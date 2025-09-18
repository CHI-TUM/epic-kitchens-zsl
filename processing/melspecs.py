import argparse
import audiofile as af
from glob import glob
import numpy as np
import os
import pandas as pd
import torch
import tqdm
import torchaudio
import torchlibrosa
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mel-spectrogram feature extraction')
    parser.add_argument(
        '--root',
        help='Path to data'
    )
    parser.add_argument(
        '--dest',
        help='Path to store features in'
    )
    parser.add_argument(
        '--use-torchlibrosa',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    
    root = args.root
    dest = args.dest
    os.makedirs(dest, exist_ok=True)
    if args.use_torchlibrosa:
        ref = 1.0
        amin = 1e-10
        top_db = None

        spectrogram = torchlibrosa.stft.Spectrogram(
            n_fft=512, 
            win_length=512, 
            hop_length=160
        )
        mel = torchlibrosa.stft.LogmelFilterBank(
            sr=16000, 
            # fmin=50, 
            # fmax=8000, 
            n_mels=64, 
            n_fft=512, 
            ref=ref, 
            amin=amin, 
            top_db=top_db
        )
        transform = lambda x: mel(spectrogram(x)).squeeze(1)
    else:
        melspect = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=512,
            hop_length=160,
            # f_min=50,
            # f_max=8000,
            n_mels=64,
        )
        transform = lambda x: melspect(x).transpose(1, 2)


    filepaths = glob(os.path.join(root, '**/*.mp3'), recursive=True)
    filenames = []
    for counter, fp in tqdm.tqdm(enumerate(filepaths), total=len(filepaths), desc='Melspects'):
        audio, fs = af.read(
            fp,
            always_2d=True
        )
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdims=True)
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)
        logmel = transform(audio)
        filename = os.path.join(args.dest, '{:012}.npy'.format(counter))
        np.save(filename, logmel)
        filenames.append(filename)

    # print(len(filepaths), len(filenames))
    filepaths = [os.path.relpath(x, root) for x in filepaths]

    features = pd.DataFrame(
        {
            'filename': filepaths, 
            'features': filenames
        }
    )
    features.to_csv(os.path.join(args.dest, 'features.csv'), index=False)
