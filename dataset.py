import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
import os, time, random, glob
from natsort import natsorted


class Hum2SongDataset(Dataset):

    def __init__(self, annotations_file, transformation, target_sampling_rate,
                 num_samples, min_amplitude):
        self.annotations = pd.read_csv(annotations_file)
        self.annotations['music_id'] = self.annotations['music_id'].astype(str)
        self.id_to_hums = self._get_id_song_dict()
        self.ids = list(self.id_to_hums.keys())
        self.transformation = transformation
        self.target_sampling_rate = target_sampling_rate
        self.num_samples = num_samples
        self.min_amplitude = min_amplitude

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        hum_path = self.annotations.iloc[index, 2]
        hum, hum_sr = torchaudio.load(hum_path)
        hum = self._preprocess(hum, hum_sr)

        song_path = self.annotations.iloc[index, 1]
        song, song_sr = torchaudio.load(song_path)
        song = self._preprocess(song, song_sr)

        music_id = self.annotations.iloc[index, 0]
        ids = [x for x in self.ids if x != music_id]
        wr_music_id = random.choice(ids)
        wr_hum_path = random.choice(self.id_to_hums[wr_music_id])
        wr_hum, sr = torchaudio.load(wr_hum_path)
        wr_hum = self._preprocess(wr_hum, sr)

        return hum, song, wr_hum

    def _get_id_song_dict(self):
        ids = list(set(self.annotations['music_id'].to_list()))
        id_to_hums = dict.fromkeys(ids)
        for _, row in self.annotations.iterrows():
            if id_to_hums[row['music_id']] is None:
                id_to_hums[row['music_id']] = [row['hum_path']]
            else:
                id_to_hums[row['music_id']].append(row['hum_path'])
        return id_to_hums

    def _preprocess(self, signal, sr):
        # resample to fixed sampling rate
        signal = Hum2SongDataset._resample_signal(signal, sr, self.target_sampling_rate)
        # stereo sound -> mono sound
        signal = Hum2SongDataset._mix_down_signal(signal)
        # resize sample to fixed length
        signal = Hum2SongDataset._resize_signal(signal, self.num_samples)
        # get mel-spectrogram features
        signal = self.transformation(signal)
        return signal

    @staticmethod
    def _resample_signal(signal, sr, target_sr=16000):
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            signal = resampler(signal)
        return signal

    @staticmethod
    def _mix_down_signal(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    @staticmethod
    def _strip_signal(signal, min_amplitude=0.004):
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

    @staticmethod
    def _resize_signal(signal, num_samples):
        if signal.shape[1] > num_samples:
            signal = Hum2SongDataset._strip_signal(signal)
            if signal.shape[1] > num_samples:
                signal = signal[:, :num_samples]
        if signal.shape[1] < num_samples:
            pad_length = num_samples - signal.shape[1]
            signal = nn.functional.pad(signal, (0, pad_length))
        return signal


class AudioDataset(Dataset):

    def __init__(self, audio_dir, transformation, target_sampling_rate, num_samples, min_amplitude):
        audio_dir = os.path.join(audio_dir, '*.mp3')
        self.audio_paths = natsorted(glob.glob(audio_dir))
        self.transformation = transformation
        self.target_sampling_rate = target_sampling_rate
        self.num_samples = num_samples
        self.min_amplitude = min_amplitude

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio, sr = torchaudio.load(audio_path)
        audio = self._preprocess(audio, sr)
        return audio

    def _preprocess(self, signal, sr):
        # resample to fixed sampling rate
        signal = Hum2SongDataset._resample_signal(signal, sr, self.target_sampling_rate)
        # stereo sound -> mono sound
        signal = Hum2SongDataset._mix_down_signal(signal)
        # get mel-spectrogram features
        signal = self.transformation(signal)
        return signal




if __name__ == "__main__":

    ANNOTATIONS_FILE = "train/train_meta.csv"
    AUDIO_DIR = "train"
    SAMPLING_RATE = 16000
    NUM_SAMPLES = SAMPLING_RATE * 8  # get 8s audio
    MIN_AMPLITUDE = 0.004

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        f_min=20,
        # f_max=22000
    )

    start = time.time()
    hum2song = Hum2SongDataset(
        ANNOTATIONS_FILE,
        mel_spectrogram,
        SAMPLING_RATE,
        NUM_SAMPLES,
        MIN_AMPLITUDE
    )
    train_loader = DataLoader(
        hum2song, 
        batch_size=4, 
        shuffle=True, 
        # num_workers=2,
        drop_last=False
    )
    print(f"There are {len(hum2song)} samples in the dataset.")

    hum, song, wr_song = next(iter(train_loader))
    print(hum.shape, song.shape, wr_song.shape)
    print(type(hum[0]), type(hum[0][0]))

    print("Time elapsed:", time.time() - start)
