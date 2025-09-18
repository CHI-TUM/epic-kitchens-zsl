import audobject
import numpy as np
import pandas as pd
import random
import torch
import typing
import os

from sklearn import preprocessing

def min_max_scaling(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    scaled_matrix = (matrix - min_value) / (max_value - min_value)
    return scaled_matrix


class Dataset(torch.utils.data.Dataset):
    r"""Torch dataset for ZSL.

    Accepts as input one DataFrame.
    Returns audio and target class.
    Optionally transforms features 
    using `transform` arguments.

    Warning: dataframe should only have
    a `label` column as metadata.
    Every other column will be considered a feature.
    """
    def __init__(
        self,
        audio: pd.DataFrame,
        transform: typing.Callable,
        label: str = 'verb_class'
    ):
        self.audio = audio
        self._audio_names = list(
            set(self.audio.columns) - set([label, 'narration'])
        )
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, item):
        target = self.audio.loc[item, self.label]
        audio = self.audio.loc[item, self._audio_names].values
        if self.transform is not None:
            audio = self.transform(audio)
        return audio.astype(np.float32), target


class LabelEncoder(audobject.Object):
    r"""Helper class to map labels."""
    def __init__(self, labels, codes=None):
        self.labels = sorted(labels)
        if codes is None:
            codes = list(range(len(labels)))
        self.codes = codes
        self.inverse_map = {code: label for code,
                    label in zip(codes, labels)}
        self.map = {label: code for code,
                            label in zip(codes, labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


class Standardizer(audobject.Object):
    r"""Helper class to normalize features."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.tolist()
        self.std = std.tolist()
        self._mean = mean
        self._std = std
    
    def encode(self, x):
        return (x - self._mean) / (self._std)

    def decode(self, x):
        return x * self._std + self._mean

    def __call__(self, x):
        return self.encode(x)


def random_split(targets, test_percentage=0.1):
    r"""Utility function used to split data.

    Accepts as input a list of targets and a percentage
    and creates three disjoint lists with targets
    to train, validate, and test on.
    """
    test_targets = random.sample(targets, int(len(targets) * test_percentage))
    other_targets = list(set(targets) - set(test_targets))

    dev_targets = random.sample(other_targets, int(len(other_targets) * test_percentage))
    train_targets = list(set(other_targets) - set(dev_targets))
    return train_targets, dev_targets, test_targets