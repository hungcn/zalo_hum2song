from torch.utils.tensorboard import SummaryWriter
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader
import os

from dataset import Hum2SongDataset
from model import ResNet


SAVE_DIR = 'checkpoints'
LOG_DIR = 'logs'
ANNOTATIONS_FILE = "train/train_meta.csv"
AUDIO_DIR = "train"
SAMPLING_RATE = 16000
MIN_AMPLITUDE = 0.004
NUM_SAMPLES = SAMPLING_RATE * 8  # get 8s audio
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCH = 100


def train(model, optimizer, batch_size, num_epoch, device="cuda"):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        f_min=20,
        # f_max=22000
    )
    train_dataset = Hum2SongDataset(
        ANNOTATIONS_FILE,
        transform,
        SAMPLING_RATE,
        NUM_SAMPLES,
        MIN_AMPLITUDE
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=False
    )
    criterion = nn.TripletMarginLoss()

    writer = SummaryWriter(LOG_DIR)

    model.train()
    step = 0
    for epoch in range(1, num_epoch + 1):
        print(f'==================== Epoch [{epoch}/{num_epoch}] ====================')

        for i, (hum, song, wr_hum) in enumerate(train_loader):
            anchor = song.to(device)
            possitive = hum.to(device)
            negative = wr_hum.to(device)

            # Train model
            emb_anc = model(anchor)
            emb_pos = model(possitive)
            emb_neg = model(negative)
            loss = criterion(emb_anc, emb_pos, emb_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss occasionally and log to tensorboard
            if i % 40 == 0:
                print('Step [{:>5d}/{:>5d}], loss: {:.4f}' 
                        .format(i, len(train_loader), loss.item()))
            writer.add_scalar("Loss", loss.item(), step)
            step += 1

        os.makedirs(SAVE_DIR, exist_ok=True)
        # Save the model checkpoints 
        if epoch % 5 == 0:
            torch.save(
                {'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, 
                os.path.join(SAVE_DIR, f'model_{epoch}.pth')
            )            

    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("------------")

    model = ResNet()
    # model.load_state_dict(torch.load('checkpoints/model_10.pth')['model'])
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train(
        model=model, 
        optimizer=optimizer,
        batch_size=BATCH_SIZE, 
        num_epoch=NUM_EPOCH, 
        device=device
    )

    

    