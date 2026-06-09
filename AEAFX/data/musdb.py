from torch.utils.data import Dataset
import torch
import torchaudio
import os
import pickle
import numpy as np
import random
import yaml
import scipy.io
from scipy.stats import qmc
import musdb
import librosa
from . import fx


def RMS(x):
    return torch.sqrt(torch.mean(torch.square(x)))


def get_effect_controls(controls_ranges):
    keys = list(controls_ranges)
    out = {}
    for key in keys:
        out[key] = controls_ranges[key][0] + np.random.rand() * (
            controls_ranges[key][1] - controls_ranges[key][0]
        )
    return out


def get_chain_controls(chain_c_ranges):
    effects = list(chain_c_ranges)
    out = {}
    for effect in effects:
        out[effect] = get_effect_controls(chain_c_ranges[effect])
    return out


class MUSDB18_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        Fx: fx.DAFx_Series,
        is_wav=False,
        random_polarity: bool = True,
        subsets="train",
        samplerate=44100,
        audio_length_s=10,
        return_effects_params=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.num_parameters = Fx.num_parameters
        self.Fx = Fx
        self.random_polarity = random_polarity
        self.samplerate = samplerate
        self.audio_length_s = audio_length_s
        self.subsets = subsets
        if subsets == "train":
            self.mus = musdb.DB(
                root=self.root_dir, subsets="train", split="train", is_wav=is_wav
            )
        if subsets == "valid":
            self.mus = musdb.DB(
                root=self.root_dir, subsets="train", split="valid", is_wav=is_wav
            )
        if subsets == "test":
            self.mus = musdb.DB(root=self.root_dir, subsets="test", is_wav=is_wav)

        self.all_controls = []

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx: int):
        track = self.mus[idx]

        max_val = 0
        while max_val < 1e-5:
            track.chunk_duration = self.audio_length_s
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            dry_waveform: np.ndarray = np.mean(track.audio, axis=1)
            max_val = np.amax(np.abs(dry_waveform))

        if np.amax(np.abs(dry_waveform)) > 1e-5:
            dry_waveform = dry_waveform / np.amax(np.abs(dry_waveform))
        else:
            dry_waveform = torch.Tensor(dry_waveform).reshape(1, -1)
            print("audio not loud enough")
            return dry_waveform, dry_waveform

        if self.random_polarity:
            polarity = np.sign(np.random.randn(1)[0])
            dry_waveform = dry_waveform * polarity

        v = np.random.rand(self.num_parameters)
        wet_waveform = self.Fx(dry_waveform, v)

        dry_waveform = torch.Tensor(dry_waveform)
        wet_waveform = torch.Tensor(wet_waveform)
        v = torch.Tensor(v)

        dry_waveform = dry_waveform.reshape(1, -1)
        wet_waveform = wet_waveform.reshape(1, -1)
        return dry_waveform, wet_waveform, v


class MUSDB18Loaded_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        Fx: fx.DAFx_Series,
        is_wav=False,
        random_polarity: bool = True,
        subsets="train",
        samplerate=44100,
        audio_length_s=10,
        return_effects_params=False,
        sample_params=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.num_parameters = Fx.num_parameters
        self.Fx = Fx
        self.random_polarity = random_polarity
        self.samplerate = samplerate
        self.audio_length_s = audio_length_s
        self.subsets = subsets
        if subsets == "train":
            self.mus = musdb.DB(
                root=self.root_dir, subsets="train", split="train", is_wav=is_wav
            )
        if subsets == "valid":
            self.mus = musdb.DB(
                root=self.root_dir, subsets="train", split="valid", is_wav=is_wav
            )
        if subsets == "test":
            self.mus = musdb.DB(root=self.root_dir, subsets="test", is_wav=is_wav)

        len_sp = int(self.audio_length_s * self.samplerate)

        self.samples_list = []
        for idx in range(len(self.mus)):
            track = self.mus[idx]
            x: np.ndarray = track.audio
            x = x.mean(axis=1)
            track_len_sp = len(x)
            num_samples_in_track = int(np.floor(track_len_sp / len_sp))
            for sp_idx in range(num_samples_in_track):
                x_sp = x[sp_idx * len_sp : (sp_idx + 1) * len_sp]
                if np.amax(np.abs(x_sp)) > 1e-3:
                    self.samples_list.append(x_sp)

        self.sample_params = sample_params
        if not self.sample_params:
            # self.all_params = np.random.rand(len(self.samples_list), self.num_parameters)
            sampler = qmc.LatinHypercube(d=self.num_parameters)
            self.all_params = sampler.random(len(self.samples_list))

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx: int):
        dry_waveform = self.samples_list[idx]

        # if np.amax(np.abs(dry_waveform)) > 1e-5:
        #     dry_waveform = dry_waveform / np.amax(np.abs(dry_waveform))
        # else:
        #     dry_waveform = torch.Tensor(dry_waveform).reshape(1, -1)
        #     print("audio not loud enough")
        #     return dry_waveform, dry_waveform

        if self.random_polarity:
            polarity = np.sign(np.random.randn(1)[0])
            dry_waveform = dry_waveform * polarity

        if self.sample_params:
            w = np.random.rand(self.num_parameters)
        else:
            w = self.all_params[idx]
        wet_waveform = self.Fx(dry_waveform, w)

        dry_waveform = torch.Tensor(dry_waveform)
        wet_waveform = torch.Tensor(wet_waveform)
        w = torch.Tensor(w)

        dry_waveform = dry_waveform.reshape(1, -1)
        wet_waveform = wet_waveform.reshape(1, -1)
        return dry_waveform, wet_waveform, w


def get_datasets(
    musdb_path: str, synFx: fx.DAFx, is_wav: bool = False, length_s: bool = 3
):
    train_dataset = MUSDB18Loaded_Dataset(
        root_dir=musdb_path,
        Fx=synFx,
        is_wav=is_wav,
        subsets="train",
        return_effects_params=True,
        audio_length_s=length_s,
        sample_params=True,
    )

    valid_dataset = MUSDB18Loaded_Dataset(
        root_dir=musdb_path,
        Fx=synFx,
        is_wav=is_wav,
        subsets="valid",
        return_effects_params=True,
        audio_length_s=length_s,
        sample_params=False,
    )

    test_dataset = MUSDB18Loaded_Dataset(
        root_dir=musdb_path,
        Fx=synFx,
        is_wav=is_wav,
        subsets="test",
        return_effects_params=True,
        audio_length_s=length_s,
        sample_params=False,
    )

    return train_dataset, valid_dataset, test_dataset
