# zalo_hum2song
Zalo AI Challenge 2021 (ZAC_2021) - Hum To Song task

## Input data structure:
```
data/
│
├── train/ - raw mp3 audios training data from ZAC
│   ├── hum
│   ├── song
│   └── train_meta.csv
│
├── train_mel/ - mel-spectrograms data after run 'prepare_meldataset' in train.py
│   ├── hum
│   ├── song
│   └── train_meta.csv
│
└── public_test/ - raw mp3 audios test data from ZAC
    ├── hum
    └── full_song
```

## Prediction:
- Put test data folder into **data/** <br>
- Change path in **test_hum_dir** and **test_song_dir** in config.yaml <br>
- Run: ```./predict.sh```

For default, prediction will run with 'public_test/hum' and 'public_test/full_song'

If run predict.sh with this docker (which have cuda installed), you should include '--gpus all'
```
e.g: sudo docker run --gpus all -v [path to data/ contains public_test]:/data -v [current dir]:/result [docker name] /bin/bash /model/predict.sh
```

## Training:
- Run: ```python train.py --prepare_mels```

