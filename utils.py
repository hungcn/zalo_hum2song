import torch, torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import os, glob
from tqdm import tqdm
import yaml



def to_melspectrogram(cfg):
    return T.MelSpectrogram(
        sample_rate=cfg['sampling_rate'],
        n_mels=cfg['n_mels'],
        n_fft=cfg['n_fft'], 
        hop_length=cfg['hop_length'],
        f_min=cfg['f_min'], f_max=cfg['f_max']
    )


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def audios_to_mels(audio_dir, save_dir, to_melspectrogram, cfg):
    _ext = cfg['audio_ext']
    print(f"Convert {_ext} audios in '{audio_dir}' to mel-spectrograms and store in '{save_dir}' ...")
    
    assert os.path.isdir(audio_dir)
    audio_paths = glob.glob(os.path.join(audio_dir, f'*.{_ext}'))
    os.makedirs(save_dir, exist_ok=True)
 
    for audio_path in tqdm(audio_paths[:20]):
        audio, sr = torchaudio.load(audio_path)
        audio = _mix_down_signal(audio)
        audio = _resample_signal(audio, sr, cfg['sampling_rate'])
        audio = _resize_signal(audio, cfg['num_samples'], cfg['min_amplitude'])
        mels = []
        for i in range(0, cfg['num_samples'] - cfg['train_segment_size'] + 1, cfg['hop_size']):
            # get mel-spectrogram features
            mel = to_melspectrogram(audio[:, i:i+cfg['train_segment_size']])
            mels.append(mel)
        batch = torch.stack(mels).squeeze(1)
        # save batch tensor for later training
        save_path = os.path.join(save_dir, os.path.basename(audio_path).replace(_ext, 'mel'))
        torch.save(batch, save_path)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0.0) 


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)   


def _resample_signal(signal, sr, target_sr=16000):
    # resample signal to target sampling rate
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)
    return signal


def _mix_down_signal(signal):
    # stereo sound -> mono sound
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def _strip_signal(signal, min_amplitude):
    signal_length = signal.shape[1]
    left = 0
    while left < signal_length and abs(signal[0][left]) < min_amplitude:
        left += 1
    right = signal_length - 1
    while right >= 0 and abs(signal[0][right]) < min_amplitude:
        right -= 1
    if left < right:
        signal = signal[:, left:right + 1]
    return signal


def _resize_signal(signal, num_samples, min_amplitude=0.0001):
    if signal.shape[1] > num_samples:
        signal = _strip_signal(signal, min_amplitude)
        if signal.shape[1] > num_samples:
            signal = signal[:, :num_samples]
    if signal.shape[1] < num_samples:
        pad_length = num_samples - signal.shape[1]
        signal = F.pad(signal, (0, pad_length))
    return signal
