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


class MedleyDBLoaded_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        Fx: fx.DAFx_Series,
        samplerate: int = 44100,
        audio_length_s: float = 10,
    ):
        super().__init__()

        audio_dir = os.path.join(root_dir, "Audio")
        songs_names_list = os.listdir(audio_dir)

        self.audio_dict = {}

        self.audio_length_sp = int(audio_length_s * samplerate)

        self.fx = Fx

        names = []

        for song_name in songs_names_list:
            filename = song_name + "_MIX.wav"

            audio, old_sr = torchaudio.load(
                os.path.join(audio_dir, song_name, filename)
            )

            audio = audio.mean(0)

            if audio.size(0) < self.audio_length_sp:
                # songs_names_list.remove(song_name)
                pass
            elif audio.square().mean() < 1e-3:
                # songs_names_list.remove(song_name)
                pass
            else:
                if old_sr != samplerate:
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=old_sr, new_freq=samplerate
                    )
                self.audio_dict[song_name] = audio.numpy()
                names.append(song_name)

        self.songs_names_list = names

    def __len__(self):
        return len(self.songs_names_list)

    def __getitem__(self, idx: int):
        key = self.songs_names_list[idx]
        dry_waveform_full = self.audio_dict[key]

        l_s = dry_waveform_full.shape[0]
        high = l_s - self.audio_length_sp

        en = 0

        while en < 1e-3:

            start_idx = np.random.randint(low=0, high=high)
            end_idx = start_idx + self.audio_length_sp

            dry_waveform = dry_waveform_full[start_idx:end_idx]
            en = np.mean(np.square(dry_waveform))

        if self.fx is not None:
            w = np.random.rand(self.fx.num_parameters)
            wet_waveform = self.fx(dry_waveform, w)
        else:
            wet_waveform = dry_waveform
            w = torch.zeros(1)

        dry_waveform = torch.Tensor(dry_waveform)
        wet_waveform = torch.Tensor(wet_waveform)
        w = torch.Tensor(w)

        x = dry_waveform.unsqueeze(0)
        y = wet_waveform.unsqueeze(0)

        x = x / x.abs().amax(dim=1, keepdim=True)
        y = y / y.abs().amax(dim=1, keepdim=True)
        return x, y, w
