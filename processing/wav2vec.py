#!/usr/bin/env python
import argparse
import json
import yaml
import os
import pandas as pd
import torch
import torchaudio
import librosa
import sys

from os import makedirs
from os.path import basename, relpath, normpath
from tqdm import tqdm
from glob import glob
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
)
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Wav2Vec features from audio segments."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing extracted audio segments (mp3)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save Wav2Vec features."
    )
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    wavs = sorted(glob(f"{INPUT_DIR}/**/*.mp3", recursive=True))
    index = list(map(lambda x: relpath(normpath(x), start=INPUT_DIR), wavs))

    vocab_dict = {}
    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json")
    tokenizer.save_pretrained("./tokenizer")

    extractors = ["facebook/wav2vec2-large-xlsr-53", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"]

    # for extractor_string in params["extractors"]:
    for extractor_string in extractors:
        try:
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                extractor_string
            )
        except OSError:
            extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )

        processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
        model = Wav2Vec2Model.from_pretrained(extractor_string).to(
            device
        )
        model.eval()


        embeddings = torch.zeros(len(index), 1024)
        for counter, wav in tqdm(enumerate(wavs)):
            audio, fs = librosa.core.load(wav, sr=None)
            audio = torch.from_numpy(audio)
            if fs != 16000:
                audio = torchaudio.transforms.Resample(fs, 16000)(audio)
            if len(audio.shape) == 2:
                audio = audio.mean(0)
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            if inputs.input_values.shape[1] < 1000:
                inputs.input_values = F.pad(inputs.input_values, (0, 1000 - inputs.input_values.shape[1]))
            with torch.no_grad():
                embeddings[counter, :] = (
                    model(
                        inputs.input_values.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                    )[0]
                    .cpu()
                    .mean(1)
                    .squeeze(0)
                )
        output_dir = os.path.join(OUTPUT_DIR, extractor_string.replace("/", "_"))
        makedirs(output_dir, exist_ok=True)
        pd.DataFrame(
            data=embeddings.numpy(),
            columns=[f"Neuron_{x}" for x in range(embeddings.shape[1])],
            index=index
        ).to_csv(f"{output_dir}/features.csv", index_label="filename")