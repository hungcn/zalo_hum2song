import torch
import torch.nn.functional as F
import torchaudio
import os, glob, time
from natsort import natsorted
from dataset import Hum2SongDataset
from model import ResNet
from tqdm import tqdm
import pandas as pd
import numpy as np


def inference_fullsong(model, audio_dir, transform, sampling_rate, segment_size, step, device):
    print("Inference fullsong ...")
    model.eval()

    audio_dir = os.path.join(audio_dir, '*.mp3')
    audio_paths = natsorted(glob.glob(audio_dir))

    for audio_path in tqdm(audio_paths):
        audio, sr = torchaudio.load(audio_path)
        # resample to fixed sampling rate
        audio = Hum2SongDataset._resample_signal(audio, sr, sampling_rate)
        # stereo sound -> mono sound
        audio = Hum2SongDataset._mix_down_signal(audio)
        # get all segments with fixed size from raw audio
        _list_emb = []
        if audio.shape[1] < segment_size:
            pad_length = segment_size - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        for i in range(0, audio.shape[1] - segment_size + 1, step):
            # get mel-spectrogram features
            mel = transform(audio[:, i:i+segment_size])
            mel = mel[None, ...].to(device)
            with torch.no_grad():
                emb = model(mel)
            _list_emb.append(emb.cpu())
        batch = torch.stack(_list_emb).squeeze(1)
        # save batch tensor for later searching
        torch.save(batch, audio_path.replace('mp3', 'pt'))


def inference_hum(model, audio_dir, transform, sampling_rate, num_samples, device):
    print("Inference hum ...")
    model.eval()

    audio_paths = os.path.join(audio_dir, '*.mp3')
    audio_paths = natsorted(glob.glob(audio_paths))

    for audio_path in tqdm(audio_paths):
        audio, sr = torchaudio.load(audio_path)
        # resample to fixed sampling rate
        audio = Hum2SongDataset._resample_signal(audio, sr, sampling_rate)
        # stereo sound -> mono sound
        audio = Hum2SongDataset._mix_down_signal(audio)
        # resize sample to fixed length
        audio = Hum2SongDataset._resize_signal(audio, num_samples, min_amplitude=0.004)
        # get mel-spectrogram features
        mel = transform(audio)
        mel = mel[None, ...].to(device)
        with torch.no_grad():
            emb = model(mel)
        batch = emb.cpu().squeeze(1)
        # save batch tensor for later searching
        torch.save(batch, audio_path.replace('mp3', 'pt'))


def search_similar_song(hum_dir, song_dir, submit_file, n_song=10, distance_type='l2'):
    print(f"Search {n_song} songs by hum ...")
    hum_paths = os.path.join(hum_dir, '*.pt')
    hum_paths = natsorted(glob.glob(hum_paths))

    song_paths = os.path.join(song_dir, '*.pt')
    song_paths = glob.glob(song_paths)
    song_names = np.array([os.path.basename(x)[:-3] for x in song_paths])
    songs = []
    for song_path in song_paths:
        songs.append(torch.load(song_path))

    df = pd.DataFrame(columns=['hum'] + [f'{i}' for i in range(n_song)])

    i = 0
    for hum_path in tqdm(hum_paths):
        hum = torch.load(hum_path)
        distances = []
        for song in songs:
            if distance_type == 'l2':
                dis = torch.cdist(hum, song).squeeze()
            elif distance_type == 'cosine':
                dis = 1 - F.cosine_similarity(hum, song)
            min_dis = torch.min(dis).item()
            distances.append(min_dis)
        distances = np.array(distances)

        sorted_ids = np.argsort(distances)
        top_songs = song_names[sorted_ids]
        hum_name = os.path.basename(hum_path).replace('pt', 'mp3')

        df.loc[i] = [hum_name] + top_songs[:n_song].tolist()
        i += 1

    df.to_csv(submit_file, index=False, header=False)



if __name__ == "__main__":

    HUM_DIR = '../public_test/hum'
    SONG_DIR = '../public_test/full_song'
    SAMPLING_RATE = 16000
    SEGMENT_SIZE = SAMPLING_RATE * 8  # 8s length
    STEP = SAMPLING_RATE // 4  # 0.25s length

    to_melspectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        f_min=20,
        # f_max=22000
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    model = ResNet()
    model.load_state_dict(torch.load('checkpoints/model_10.pth')['model'])
    model = model.to(device)

    inference_fullsong(
        model=model,
        audio_dir=SONG_DIR,
        transform=to_melspectrogram,
        sampling_rate=SAMPLING_RATE,
        segment_size=SEGMENT_SIZE,
        step=STEP,
        device=device
    )
    
    inference_hum(
        model=model,
        audio_dir=HUM_DIR,
        transform=to_melspectrogram,
        sampling_rate=SAMPLING_RATE,
        num_samples=SEGMENT_SIZE,
        device=device
    )

    search_similar_song(
        hum_dir=HUM_DIR,
        song_dir=SONG_DIR,
        submit_file='submit.csv',
        n_song=10,
        distance_type='l2'
    )

