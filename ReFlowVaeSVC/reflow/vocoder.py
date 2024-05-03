import os
import yaml
import torch
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False
import torch.nn as nn
import numpy as np
from ReFlowVaeSVC.nsf_hifigan.nvSTFT import STFT
from ReFlowVaeSVC.nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from .reflow import Bi_RectifiedFlow
from .naive_v2_diff import NaiveV2Diff
from .wavenet import WaveNet

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

        
def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model    
    if args.model.type == 'RectifiedFlow_VAE':
        model = Unit2Wav_VAE(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    args.model.back_bone,
                    args.model.use_attention)
                    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
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
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
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
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Wav_VAE(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6, 
            n_chans=512,
            n_hidden=256,
            back_bone='lynxnet',
            use_attention=False):
        super().__init__()
        self.f0_embed = nn.Linear(1, n_hidden)
        self.use_attention = use_attention
        if use_attention:
            self.unit_embed = nn.Linear(n_unit, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            self.attention = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=0.1,
                    activation='gelu',
                ),
                nn.Linear(n_hidden, out_dims),
            )
        else:
            self.unit_embed = nn.Linear(n_unit, out_dims)
            self.volume_embed = nn.Linear(1, out_dims)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        if back_bone is None or back_bone == 'lynxnet':        
            self.reflow_model = Bi_RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=n_hidden, use_mlp=False))
        elif back_bone == 'wavenet':
            self.reflow_model = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))
        else:
            raise ValueError(f" [x] Unknown Backbone: {back_bone}")
            
    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # condition
        cond = self.f0_embed((1+ f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    cond = cond + v * self.spk_embed(spk_id_torch - 1)
            else:
                cond = cond + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift / 5)
        
        # vae mean
        x = self.unit_embed(units) + self.volume_embed(volume)
        if self.use_attention:
            x = self.attention(x)
        
        # vae noise
        x += torch.randn_like(x)
        
        x = self.reflow_model(infer=infer, x_start=x, x_end=gt_spec, cond=cond, infer_step=infer_step, method='euler', use_tqdm=True)
        
        if return_wav and infer:
            return vocoder.infer(x, f0)
        else:
            return x
            
    def vae_infer(self, input_mel, input_f0, input_spk_id, output_f0, output_spk_id=None, spk_mix_dict=None, aug_shift=None,
            infer_step=10, method='euler'):
        
        # source condition
        source_cond = self.f0_embed((1+ input_f0 / 700).log()) + self.spk_embed(input_spk_id - 1)
        
        # target condition
        target_cond = self.f0_embed((1+ output_f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    target_cond = target_cond + v * self.spk_embed(spk_id_torch - 1)
            else:
                target_cond = target_cond + self.spk_embed(output_spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            target_cond = target_cond + self.aug_shift_embed(aug_shift / 5)
            
        print("\nExtracting features...")
        latent = self.reflow_model(infer=True, x_end=input_mel, cond=source_cond, infer_step=infer_step, method='euler', use_tqdm=True)
        print("\nSynthesizing...")
        output_mel = self.reflow_model(infer=True, x_start=latent, cond=target_cond, infer_step=infer_step, method='euler', use_tqdm=True)
        return output_mel
