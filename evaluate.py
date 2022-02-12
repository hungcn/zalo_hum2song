import torch
import os, glob, shutil
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from model import ResNet


parser = argparse.ArgumentParser()
parser.add_argument("--train_mel_dir", default='../data/train_mel')
parser.add_argument("--val_mel_dir", default='../data/val_mel')
parser.add_argument("--model_name", default='resnet18')
parser.add_argument("--checkpoint", default='./checkpoints/base/model.pth')
parser.add_argument("--distance", choices=['cosine', 'l2', 'dot'], default='cosine')



def split_train_to_eval(train_mel_dir, val_mel_dir, N=600):
    if os.path.isdir(val_mel_dir):
        return
    print(f'\nRandomly get {N} hums from training set to make validation set ...')
    np.random.seed(0)
    os.makedirs(val_mel_dir)

    train_meta_file = os.path.join(train_mel_dir, 'train_meta.csv')
    df = pd.read_csv(train_meta_file)[['hum_path', 'music_id']]
    assert N < len(df)
    test_df = df.sample(n=N, replace=False)
    test_df['hum_path'] = test_df['hum_path'].apply(lambda x: os.path.basename(x).replace('mp3', 'mel'))
    
    eval_meta_file = os.path.join(val_mel_dir, 'val_meta.csv')
    test_df.to_csv(eval_meta_file, index=False)

    eval_hum_dir =  os.path.join(val_mel_dir, 'hum')
    eval_song_dir =  os.path.join(val_mel_dir, 'song')
    train_hum_dir = os.path.join(train_mel_dir, 'hum')
    train_song_dir = os.path.join(train_mel_dir, 'song')

    os.makedirs(eval_hum_dir, exist_ok=True)
    os.makedirs(eval_song_dir, exist_ok=True)

    for _, row in tqdm(test_df.iterrows(), desc='copy hum mels'):
        src_hum = os.path.join(train_hum_dir, row['hum_path'])
        shutil.copy(src_hum, eval_hum_dir)
    
    music_ids = set()
    for _, row in tqdm(df.iterrows(), desc='copy music mels'):
        if row['music_id'] in music_ids:
            continue
        music_ids.add(row['music_id'])
        name = os.path.basename(row['hum_path']).replace('mp3', 'mel')
        src_song = os.path.join(train_song_dir, name)
        dst_song = os.path.join(eval_song_dir, f"{row['music_id']}.mel")
        shutil.copyfile(src_song, dst_song)


def evaluate(model, val_mel_dir, n_song=10, distance_type='cosine'):
    print(f"\n===== Evaluation by MRR@{n_song}")
    model.eval()

    assert os.path.isdir(val_mel_dir)
    hum_dir = os.path.join(val_mel_dir, 'hum/*.mel')
    song_dir = os.path.join(val_mel_dir, 'song/*.mel')
    meta_file = os.path.join(val_mel_dir, 'val_meta.csv')

    hum_paths = glob.glob(hum_dir)
    hum_names = np.array([os.path.basename(x) for x in hum_paths])
    hums = []
    for hum_path in tqdm(hum_paths, desc='Inference hums'):
        mel = torch.load(hum_path)
        mel = mel.unsqueeze(1).to(device)
        with torch.no_grad():
            hum = model(mel)
        hums.append(hum.cpu())

    song_paths = glob.glob(song_dir)
    song_names = np.array([int(os.path.basename(x)[:-4]) for x in song_paths])
    songs = []
    for song_path in tqdm(song_paths, desc='Inference songs'):
        mel = torch.load(song_path)
        mel = mel.unsqueeze(1).to(device)
        with torch.no_grad():
            song = model(mel)
        songs.append(song.cpu())

    df = pd.read_csv(meta_file)
    N = len(hums)

    score = 0.0
    for i in tqdm(range(N), desc='Compute'):
        hum_name = hum_names[i]
        hum = hums[i]
        distances = []
        for song in songs:           
            if distance_type == 'cosine':
                dis = 1 - F.cosine_similarity(hum, song)
            elif distance_type == 'l2':
                dis = (hum - song).pow(2).sum(dim=1).sqrt()
            elif distance_type == 'dot':
                dis = -torch.sum(hum * song, dim=1)
            else:
                dis = torch.abs(hum - song).sum(dim=1)
            dis = torch.sum(dis).item()
            distances.append(dis)
        distances = np.array(distances)
        sorted_ids = np.argsort(distances)
        top_songs = song_names[sorted_ids]
        top_songs = top_songs[:n_song]

        song_name = df[df['hum_path'] == hum_name]['music_id'].values[0]
        idx = np.where(top_songs == song_name)[0]
        if len(idx) > 0:
            score += 1.0 / (idx[0] + 1)

    print("--> Score:", score / N)




if __name__ == "__main__":

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    split_train_to_eval(
        train_mel_dir=args.train_mel_dir,
        val_mel_dir=args.val_mel_dir
    )

    model = ResNet(model_name=args.model_name)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = torch.jit.script(model).to(device)

    evaluate(
        model=model, 
        val_mel_dir=args.val_mel_dir,
        distance_type=args.distance
    )
