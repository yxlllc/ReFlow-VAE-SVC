import numpy as np
import torch
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False
import torch.nn.functional as F
import pyworld as pw
import parselmouth
import torchcrepe
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from fairseq import checkpoint_utils
from ReFlowVaeSVC.encoder.hubert.model import HubertSoft
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample

CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}


def MaskedAvgPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)


def MedianPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]


class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL :
                from ReFlowVaeSVC.encoder.rmvpe.inference import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
        if f0_extractor == 'fcpe':
            if torch.cuda.is_available():
                self.device_fcpe = 'cuda'
            elif use_torch_musa:
                if torch.musa.is_available():
                    self.device_fcpe = 'musa'
                else:
                    self.device_fcpe = 'cpu'
            else:
                self.device_fcpe = 'cpu'
            if 'fcpe' not in F0_KERNEL :
                from torchfcpe import spawn_bundled_infer_model
                F0_KERNEL['fcpe'] = spawn_bundled_infer_model(device=self.device_fcpe)
            self.fcpe = F0_KERNEL['fcpe']
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
        
        # extract f0 using parselmouth
        if self.f0_extractor == 'parselmouth':
            l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
            r_pad = int(self.hop_size * ((len(audio) - 1) // self.hop_size + 1) - len(audio) + l_pad + 1)
            s = parselmouth.Sound(np.pad(audio, (l_pad, r_pad)), self.sample_rate).to_pitch_ac(
                time_step = self.hop_size / self.sample_rate, 
                voicing_threshold = 0.6,
                pitch_floor = self.f0_min, 
                pitch_ceiling = self.f0_max)
            assert np.abs(s.t1 - 1.5 / self.f0_min) < 0.001
            f0 = np.pad(s.selected_array['frequency'], (start_frame, 0))
            if len(f0) < n_frames:
                f0 = np.pad(f0, (0, n_frames - len(f0)))
            f0 = f0[: n_frames]
            
        # extract f0 using dio
        elif self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                channels_in_octave=2, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using harvest
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using crepe        
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))
            
            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)
            pd = MedianPool1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = MaskedAvgPool1d(f0, 4)
            
            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array([f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))
        
        # extract f0 using rmvpe
        elif self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
        
        # extract f0 using fcpe
        elif self.f0_extractor == "fcpe":
            _audio = torch.from_numpy(audio).to(self.device_fcpe).unsqueeze(0)
            f0 = self.fcpe(_audio, sr=self.sample_rate, decoder_mode="local_argmax", threshold=0.006)
            f0 = f0.squeeze().cpu().numpy()
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = np.array([np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume
    
         
class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None,
                 cnhubertsoft_gate=10):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif use_torch_musa:
                if torch.musa.is_available():
                    device = 'musa'
                else:
                    device = 'cpu'
            else:
                device = 'cpu'
        self.device = device
        
        is_loaded_encoder = False
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
            is_loaded_encoder = True
        if encoder == 'hubertbase':
            self.model = Audio2HubertBase(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768':
            self.model = Audio2HubertBase768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768l12':
            self.model = Audio2HubertBase768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertlarge1024l24':
            self.model = Audio2HubertLarge1024L24(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec':
            self.model = Audio2ContentVec(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768':
            self.model = Audio2ContentVec768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12':
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'cnhubertsoftfish':
            self.model = CNHubertSoftFish(encoder_ckpt, device=device, gate_size=cnhubertsoft_gate)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned
        
class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()
     
    def forward(self, 
                audio): # B, T
        with torch.inference_mode():  
            units = self.hubert.units(audio.unsqueeze(1))
            return units


class Audio2ContentVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = self.hubert.final_proj(logits[0])
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units    


class CNHubertSoftFish(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
        super().__init__()
        self.device = device
        self.gate_size = gate_size

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./pretrain/TencentGameMate/chinese-hubert-base")
        self.model = HubertModel.from_pretrained("./pretrain/TencentGameMate/chinese-hubert-base")
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
        # self.label_embedding = nn.Embedding(128, 256)

        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, audio):
        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        return self._forward(input_values[0])

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # Top-k gating
        topk, indices = torch.topk(features, self.gate_size, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)

        return features.to(self.device)  # .transpose(1, 2)

    
class Audio2HubertBase():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = self.hubert.final_proj(logits[0])
            return units


class Audio2HubertBase768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units


class Audio2HubertBase768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 12,  # layer 12
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units


class Audio2HubertLarge1024L24():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 24,  # layer 24
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units
