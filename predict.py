import torch, torchaudio
import torch.nn.functional as F
import os, glob
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool

from model import ResNet
import utils
    


def prepare_embedddings(audio_dir, song: bool, model, cfg, to_melspectrogram, device):
    model.eval()

    audio_dir = os.path.join(audio_dir, f"*.{cfg['audio_ext']}")
    audio_paths = glob.glob(audio_dir)

    for audio_path in tqdm(audio_paths):
        audio, sr = torchaudio.load(audio_path)
        # preprocessing audio signal
        audio = utils._mix_down_signal(audio)
        audio = utils._resample_signal(audio, sr, cfg['sampling_rate'])
        audio = audio[:, :cfg['max_num_samples']]
        if song == False or (song and audio.size(1) < cfg['num_samples']):
            audio = utils._resize_signal(audio, cfg['num_samples'], cfg['min_amplitude'])
        # get all segments with fixed size from audio
        _list_emb = []
        for i in range(0, audio.size(1) - cfg['test_segment_size'] + 1, cfg['hop_size']):
            # get mel-spectrogram features
            mel = to_melspectrogram(audio[:, i:i+cfg['test_segment_size']])
            mel = mel[None, ...].to(device)
            with torch.no_grad():
                emb = model(mel)
            _list_emb.append(emb.cpu())
        batch = torch.stack(_list_emb).squeeze(1)
        # save batch tensor for later searching
        torch.save(batch, audio_path.replace(cfg['audio_ext'], 'pt'))
  

def search_similar_song(hum_dir, song_dir, submit_file, n_songs=10, distance_type='cosine', _ext='mp3'):
    hum_paths = os.path.join(hum_dir, '*.pt')
    hum_paths = natsorted(glob.glob(hum_paths))

    song_paths = os.path.join(song_dir, '*.pt')
    song_paths = list(glob.glob(song_paths))
    song_names = np.array([os.path.basename(x)[:-3] for x in song_paths])
    songs = []
    for song_path in song_paths:
        songs.append(torch.load(song_path))

    df = pd.DataFrame(columns=['hum'] + [f'{i}' for i in range(n_songs)])

    i = 0
    for hum_path in tqdm(hum_paths):
        hum = torch.load(hum_path)
        n_seg = hum.size(0)
        hum = hum.reshape(1,-1)
        distances = []
        for song in songs:   
            dis = float('inf')
            for j in range(0, song.size(0) - n_seg + 1, 1):
                window = song[j:j+n_seg].reshape(1,-1)
                if distance_type == 'cosine':
                    d = 1 - F.cosine_similarity(hum, window)
                elif distance_type == 'dot':
                    d = -torch.sum(hum * window, dim=1)
                elif distance_type == 'l2':
                    d = (hum - window).pow(2).sum(dim=1) 
                dis = min(dis, d.item())
            distances.append(dis)
        distances = np.array(distances)
        sorted_ids = np.argsort(distances)
        top_songs = song_names[sorted_ids]
        top_songs = top_songs[:n_songs]
        hum_name = os.path.basename(hum_path).replace('pt', _ext)

        df.loc[i] = [hum_name] + top_songs.tolist()
        i += 1

    if not os.path.isdir(os.path.dirname(submit_file)):
        os.makedirs(os.path.dirname(submit_file))
    
    df.to_csv(submit_file, index=False, header=False)
    print("Saved to", submit_file)


def clean_embeddings(hum_dir, song_dir):
    hum_paths = os.path.join(hum_dir, '*.pt')
    song_paths = os.path.join(song_dir, '*.pt')

    hum_paths = glob.glob(hum_paths)
    song_paths = glob.glob(song_paths)
    paths = hum_paths + song_paths

    with Pool() as pool:
        pool.map(os.remove, paths)



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    cfg = utils.read_yaml('config.yaml')

    HUM_DIR = os.path.join(cfg['root'], cfg['test_hum_dir'])
    SONG_DIR = os.path.join(cfg['root'], cfg['test_song_dir'])

    to_melspectrogram = utils.to_melspectrogram(cfg)

    model = ResNet(embed_dim=cfg['embed_dim'], backbone=cfg['backbone'])
    model.load_state_dict(torch.load(cfg['checkpoint_path'])['model'])
    model = torch.jit.script(model).to(device)

    print("Inference fullsong ...")
    prepare_embedddings(
        audio_dir=SONG_DIR,
        song=True,
        model=model, cfg=cfg, 
        to_melspectrogram=to_melspectrogram, 
        device=device
    )
    print("Inference hum ...")
    prepare_embedddings(
        audio_dir=HUM_DIR,
        song=False,
        model=model, cfg=cfg, 
        to_melspectrogram=to_melspectrogram, 
        device=device
    )

    print("Predicting ...")
    search_similar_song(
        hum_dir=HUM_DIR,
        song_dir=SONG_DIR,
        submit_file='result/submission.csv',
        n_songs=cfg['n_songs'],
        distance_type=cfg['distance_type'],
        _ext=cfg['audio_ext']
    )

    print("Clean embeddings ...")
    clean_embeddings(HUM_DIR, SONG_DIR)

    print("Finish prediction.")

