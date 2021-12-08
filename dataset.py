import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import random, os, glob
from tqdm import tqdm


ANNOTATIONS_FILE = "../train/train_meta.csv"
ROOT = '../'
SAMPLING_RATE = 16000
NUM_SAMPLES = SAMPLING_RATE * 8  # 8s
SEGMENT_SIZE = SAMPLING_RATE * 2  # 2s
STEP = 16000  # 1s
MIN_AMPLITUDE = 0.0001


def _resample_signal(signal, sr, target_sr=16000):
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)
    return signal


def _mix_down_signal(signal):
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
        signal = nn.functional.pad(signal, (0, pad_length))
    return signal

class Hum2SongDataset(Dataset):
    "MelSpectrogram dataset for Hum-To-Song task"
    def __init__(self, root, annotations_file, transform):
        super().__init__()
        df = pd.read_csv(annotations_file)
        df['song_path'] = df['song_path'].apply(lambda x: os.path.join(root, x).replace('mp3', 'mel'))
        df['hum_path'] = df['hum_path'].apply(lambda x: os.path.join(root, x).replace('mp3', 'mel'))
        self.annotations = df
        self.ids = sorted(list(set(self.annotations['music_id'].to_list())))
        self.transform = transform
        self.id_to_label = {music_id: i for i, music_id in enumerate(self.ids)}

    def _get_song_hum_by_id(self, music_id : int):
        rows = self.annotations[self.annotations['music_id'] == music_id]
        pairs = rows.values[:, 1:].tolist()
        return pairs
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        hum_path = self.annotations.iloc[index, 2]
        hum = torch.load(hum_path)
        
        song_path = self.annotations.iloc[index, 1]
        song = torch.load(song_path)

        music_id = self.annotations.iloc[index, 0]

        # wr_ids = [x for x in self.ids if x != music_id]
        # wr_music_id = random.choice(wr_ids)
        # wr_pairs = self._get_song_hum_by_id(wr_music_id)
        # _, wr_hum_path = random.choice(wr_pairs)
        # wr_hum = torch.load(wr_hum_path)

        if self.transform:
            hum = self.transform(hum)
            song = self.transform(song)
            # wr_hum = self.transform(wr_hum)

        return hum, song, self.id_to_label[music_id]


def audios_to_mels(audio_dir, save_dir, to_melspectrogram, 
                    target_sr=SAMPLING_RATE,
                    num_samples=NUM_SAMPLES,
                    segment_size=SEGMENT_SIZE,
                    step=STEP,
                    min_amplitude=MIN_AMPLITUDE):

    _audio_ext = 'mp3'
    print(f"Convert {_audio_ext} audios in '{audio_dir}' to mel-spectrograms and store in '{save_dir}' ...")
    
    assert os.path.isdir(audio_dir)
    audio_paths = glob.glob(os.path.join(audio_dir, f'*.{_audio_ext}'))
    os.makedirs(save_dir, exist_ok=True)

    for audio_path in tqdm(audio_paths):
        audio, sr = torchaudio.load(audio_path)
        # stereo -> mono
        audio = _mix_down_signal(audio)
        # resample to target sampling rate
        audio = _resample_signal(audio, sr, target_sr)
        # resize signal to fixed length
        audio = _resize_signal(audio, num_samples, min_amplitude)
        mels = []
        for i in range(0, num_samples - segment_size + 1, step):
            # get mel-spectrogram features
            mel = to_melspectrogram(audio[:, i:i+segment_size])
            mels.append(mel)
        batch = torch.stack(mels).squeeze(1)
        # save batch tensor for later training
        save_path = os.path.join(save_dir, os.path.basename(audio_path).replace(f'{_audio_ext}', 'mel'))
        torch.save(batch, save_path)


if __name__ == "__main__":

    to_melspectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000
    )

    audios_to_mels(
        audio_dir='../train_mp3/hum',
        save_dir='../train_mel_80x251/hum',
        to_melspectrogram=to_melspectrogram
    )

    audios_to_mels(
        audio_dir='../train_mp3/song',
        save_dir='../train_mel_80x251/song',
        to_melspectrogram=to_melspectrogram
    )
