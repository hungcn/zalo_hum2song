from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import os
import pandas as pd

from dataset import MelDataset
from model import ResNet
import utils
from pytorch_metric_learning import losses 



def prepare_meldataset(cfg, to_melspectrogram):
    train_hum_dir = os.path.join(cfg['root'], cfg['train_hum_dir'])
    train_song_dir = os.path.join(cfg['root'], cfg['train_song_dir'])

    if os.path.isdir(train_hum_dir) and os.path.isdir(train_song_dir):
        return

    df = pd.read_csv(os.path.join(cfg['root'], cfg['annotation_file']))
    df['song_path'] = df['song_path'].apply(
        lambda x: x.replace(cfg['audio_ext'], 'mel').replace(cfg['song_dir'], cfg['train_song_dir'])
    )
    df['hum_path'] = df['hum_path'].apply(
        lambda x: x.replace(cfg['audio_ext'], 'mel').replace(cfg['hum_dir'], cfg['train_hum_dir'])
    )
    meta_file = os.path.join(cfg['root'], cfg['train_meta'])
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    df.to_csv(meta_file, index=False)

    utils.audios_to_mels(
        audio_dir=os.path.join(cfg['root'], cfg['hum_dir']),
        save_dir=train_hum_dir,
        to_melspectrogram=to_melspectrogram,
        cfg=cfg
    )
    utils.audios_to_mels(
        audio_dir=os.path.join(cfg['root'], cfg['song_dir']),
        save_dir=train_song_dir,
        to_melspectrogram=to_melspectrogram,
        cfg=cfg
    )


def train(model, optimizer, start_epoch, cfg, device="cuda"):
    transform = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=10),
        T.TimeMasking(time_mask_param=20)
    )
    train_dataset = MelDataset(cfg, transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=False
    )
    criterion = losses.MultiSimilarityLoss()

    writer = SummaryWriter(cfg['log_dir'])

    model.train()
    step = (start_epoch-1) * len(train_loader)

    for epoch in range(start_epoch, cfg['num_epochs'] + 1):
        print(f"==================== Epoch [{epoch}/{cfg['num_epochs']}] ====================")
        for i, (mels, labels) in enumerate(train_loader):
            mels = mels.reshape(-1, 1, mels.size(-2), mels.size(-1))
            labels = labels.reshape(-1).to(device)

            mels = mels.to(device)
            embeddings = model(mels)
            
            # Train model
            loss = criterion(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss occasionally and log to tensorboard
            if i % 40 == 0:
                print('Step [{:>5d}/{:>5d}], loss: {:.4f}' 
                        .format(i, len(train_loader), loss.item()))
            writer.add_scalar("Loss", loss.item(), step)
            step += 1

        os.makedirs(cfg['save_dir'], exist_ok=True)
        # Save the model checkpoints 
        if epoch % 10 == 0:
            torch.save(
                {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, 
                os.path.join(cfg['save_dir'], f'model_{epoch}.pth')
            )      
      
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    cfg = utils.read_yaml('config.yaml')

    to_melspectrogram = utils.to_melspectrogram(cfg)
    print("Preparing mels dataset from audios ...")
    prepare_meldataset(cfg, to_melspectrogram)

    model = ResNet(embed_dim=cfg['embed_dim'])
    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    start_epoch = 1

    if cfg['checkpoint_path'] is None:
        start_epoch = 1
        model.apply(utils.weights_init)
    else:
        checkpoint = torch.load(cfg['checkpoint_path'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    model = torch.jit.script(model).to(device)
    utils.optimizer_to(optimizer, device)

    train(
        model=model, 
        optimizer=optimizer, 
        start_epoch=start_epoch, 
        cfg=cfg,
        device=device
    )

    

    