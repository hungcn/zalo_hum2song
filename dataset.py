import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class MelDataset(Dataset):
    "MelSpectrogram dataset for ZaloAI2021 task Hum-To-Song"
    def __init__(self, cfg, transform=None):
        super().__init__()
        df = pd.read_csv(os.path.join(cfg['root'], cfg['train_meta']))
        df['song_path'] = df['song_path'].apply(lambda x: os.path.join(cfg['root'], x))
        df['hum_path'] = df['hum_path'].apply(lambda x: os.path.join(cfg['root'], x))
        self.df = df
        self.ids = sorted(list(set(self.df['music_id'].to_list())))
        self.transform = transform
        self.id_to_label = {music_id: i for i, music_id in enumerate(self.ids)}

    def _get_song_hum_by_id(self, music_id : int):
        rows = self.df[self.df['music_id'] == music_id]
        pairs = rows.values[:, 1:].tolist()
        return pairs
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        hum_path = self.df.iloc[index, 2]
        song_path = self.df.iloc[index, 1]

        hum = torch.load(hum_path)
        song = torch.load(song_path)
        if self.transform:
            hum = self.transform(hum)
            song = self.transform(song)
        mels = torch.cat([song, hum])        

        music_id = self.df.iloc[index, 0]
        label = self.id_to_label[music_id]
        label = torch.ones(hum.size(0), dtype=torch.int64) * label
        labels = torch.cat([label, label])
    
        return mels, labels



if __name__ == "__main__":

    import utils
    cfg = utils.read_yaml('config.yaml')

    data = MelDataset(cfg)
    from torch.utils.data import DataLoader

    loader = DataLoader(data, batch_size=2)
    mels, labels = next(iter(loader))
    print(mels.shape)
    print(labels.shape)
