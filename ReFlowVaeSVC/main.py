import os
import torch
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False
import librosa
import argparse
import numpy as np
import soundfile as sf
import hashlib
import torch.nn.functional as F
from ast import literal_eval
from ReFlowVaeSVC.slicer import Slicer
from ReFlowVaeSVC.reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
from ReFlowVaeSVC.reflow.vocoder import load_model_vocoder
from tqdm import tqdm


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_ckpt",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu/cuda/musa, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-sid",
        "--source_spk_id",
        type=str,
        required=False,
        default='none',
        help="source speaker id (for multi-speaker model) | default: none",
    )
    parser.add_argument(
        "-tid",
        "--target_spk_id",
        type=str,
        required=False,
        default=1,
        help="target speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-step",
        "--infer_step",
        type=str,
        required=False,
        default='auto',
        help="sample steps | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="euler or rk4 | default: auto",
    )
    return parser.parse_args(args=args, namespace=namespace)


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]
    return signal.permute(0, 2, 1)


def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    #device = 'cpu' 
    device = cmd.device
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
    
    # load reflow model
    model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)
    
    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    
    # get MD5 hash from wav file
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)
    
    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")
    
    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
        # f0 cache load
        print('Loading pitch curves for input audio from cache directory...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        # extract f0
        print('Pitch extractor type: ' + cmd.pitch_extractor)
        pitch_extractor = F0_Extractor(
                            cmd.pitch_extractor, 
                            sample_rate, 
                            hop_size, 
                            float(cmd.f0_min), 
                            float(cmd.f0_max))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
        
        # f0 cache save
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)
    
    # key change
    input_f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    output_f0 = input_f0 * 2 ** (float(cmd.key) / 12)
    
    # formant change
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)
    
    # source speaker id
    if cmd.source_spk_id == 'none':
        # load units encoder
        if args.data.encoder == 'cnhubertsoftfish':
            cnhubertsoft_gate = args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
            units_encoder = Units_Encoder(
                                args.data.encoder, 
                                args.data.encoder_ckpt, 
                                args.data.encoder_sample_rate, 
                                args.data.encoder_hop_size,
                                cnhubertsoft_gate=cnhubertsoft_gate,
                                device = device)
        # extract volume 
        print('Extracting the volume envelope of the input audio...')
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
        
    else:
        source_spk_id = torch.LongTensor(np.array([[int(cmd.source_spk_id)]])).to(device)
        print('Using VAE mode...')
        print('Source Speaker ID: '+ str(int(cmd.source_spk_id)))
    
    # targer speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    target_spk_id = torch.LongTensor(np.array([[int(cmd.target_spk_id)]])).to(device)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('Target Speaker ID: '+ str(int(cmd.target_spk_id)))
    
    # sampling method    
    if cmd.method == 'auto':
        method = args.infer.method
    else:
        method = cmd.method
        
    # infer step
    if cmd.infer_step == 'auto':
        infer_step = args.infer.infer_step
    else:
        infer_step = int(cmd.infer_step)

    if infer_step < 0:
        print('infer step cannot be negative!')
        exit(0)
    
    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            if cmd.source_spk_id == 'none':
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
                seg_f0 = output_f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
                
                seg_output = model(
                    seg_units, 
                    seg_f0, 
                    seg_volume, 
                    spk_id = target_spk_id, 
                    spk_mix_dict = spk_mix_dict,
                    aug_shift = formant_shift_key,
                    vocoder=vocoder,
                    infer=True,
                    return_wav=True,
                    infer_step=infer_step, 
                    method=method)
                seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            else:           
                seg_input_mel = vocoder.extract(seg_input, sample_rate)
                seg_input_mel = torch.cat((seg_input_mel, seg_input_mel[:,-1:,:]), 1)
                seg_input_f0 = input_f0[:, start_frame : start_frame + seg_input_mel.size(1), :]
                seg_output_f0 = output_f0[:, start_frame : start_frame + seg_input_mel.size(1), :]

                seg_output_mel = model.vae_infer(
                                    seg_input_mel, 
                                    seg_input_f0,
                                    source_spk_id,
                                    seg_output_f0,
                                    target_spk_id,
                                    spk_mix_dict,
                                    formant_shift_key,
                                    infer_step, 
                                    method)
                seg_output = vocoder.infer(seg_output_mel, seg_output_f0)
            
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
        sf.write(cmd.output, result, args.data.sampling_rate)
    