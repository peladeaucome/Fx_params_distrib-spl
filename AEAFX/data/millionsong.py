import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy
import csv
from . import fx
import os
import torchaudio
import numpy as np
import json
import librosa
from .audio_utils import read_wav
import scipy.io.wavfile

class MillionSong(Dataset):
    def __init__(
        self,
        ds_path: str,
        Fx: fx.DAFx_Series,
        samplerate: int = 22050,
        len_s: float = 5,
        dataset_length: int = None,
    ):
        super().__init__()
        self.ds_path = ds_path
        self.num_parameters = Fx.num_parameters
        self.Fx = Fx
        self.samplerate = samplerate
        self.len_s = len_s
        self.len_sp = int(len_s * samplerate)

        self.audio_path_list = []

        n = 0
        with open(os.path.join(ds_path, "metadata.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row["file_path"]
                full_audio_path = os.path.join(ds_path, audio_path)

                md = torchaudio.info(full_audio_path)
                x, sr = torchaudio.load(full_audio_path)

                x = x.mean(0)

                len_file_sp = md.num_frames
                if len_file_sp > self.len_sp and x.std().item() > 1e-2:
                    self.audio_path_list.append(row["file_path"])
                    n += 1
                if dataset_length is not None:
                    if n >= dataset_length:
                        break

    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        audio_path: str = self.audio_path_list[idx]
        full_audio_path = os.path.join(self.ds_path, audio_path)

        md = torchaudio.info(full_audio_path)

        file_length = md.num_frames

        try_again: bool = True
        while try_again:
            start_sp = np.random.randint(low=0, high=file_length - self.len_sp)
            x, sr = torchaudio.load(
                full_audio_path, frame_offset=start_sp, num_frames=self.len_sp
            )
            x = x.mean(0)

            if x.std() > 1e-2:
                try_again = False

        x = x.numpy()

        v = np.random.rand(self.num_parameters)
        y = self.Fx(x, v)

        x = torch.Tensor(x).view(1, self.len_sp)
        y = torch.Tensor(y).view(1, self.len_sp)
        v = torch.Tensor(v)

        # print(x.std(), y.std(), v.std())
        return x, y, v


class MillionSong_2(Dataset):
    def __init__(
        self,
        ds_path: str,
        Fx: fx.DAFx_Series,
        Fx_norm: fx.DAFx_Series = None,
        samplerate: int = 22050,
        len_s: float = 5,
    ):
        super().__init__()
        self.ds_path = ds_path
        self.num_parameters = Fx.num_parameters
        self.Fx_norm = Fx_norm
        self.Fx = Fx
        self.samplerate = samplerate
        self.len_s = len_s
        self.len_sp = int(len_s * samplerate)

        self.audio_path_list = []

        n = 0
        with open(os.path.join(ds_path, "audio_path_list.json"), "r") as fp:
            self.audio_path_list = json.load(fp)

    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        audio_path: str = self.audio_path_list[idx]
        full_audio_path = os.path.join(self.ds_path, audio_path)

        # md = torchaudio.info(full_audio_path)

        # file_length = md.num_frames

        try_again: bool = True
        while try_again:
            sr, x = scipy.io.wavfile.read(full_audio_path)
            # x=x.T
            x=x/np.amax(np.abs(x))
            # print(x.shape)
            file_length=x.shape[0]
            start_sp = np.random.randint(low=0, high=file_length - self.len_sp)
            x=x[start_sp:start_sp+self.len_sp]
            # x, sr = torchaudio.load(
            #     full_audio_path, frame_offset=start_sp, num_frames=self.len_sp
            # )

            # x, sr = read_wav(
            #     full_audio_path,
            #     seek_time=start_sp / self.samplerate,
            #     duration=self.len_s,
            # )

            if self.samplerate != sr:
                x = torch.Tensor(
                    librosa.resample(
                        y=x.numpy(), orig_sr=sr, target_sr=self.samplerate, axis=1
                    )
                ).to(x)

            # x = x.mean(0)

            if x.std() > 1e-2:
                try_again = False

            # x=x/x.std()

        # x = x.numpy()
        if self.Fx_norm is not None:
            x = self.Fx_norm(x, np.zeros((1, 1)))
        v = np.random.rand(self.num_parameters)
        y = self.Fx(x, v)

        x = torch.Tensor(x).unsqueeze(0)
        y = torch.Tensor(y).unsqueeze(0)
        v = torch.Tensor(v)

        # print(x.std(), y.std(), v.std())
        return x, y, v


class MillionSong_2_norm(Dataset):
    def __init__(
        self,
        ds_path: str,
        Fx: fx.DAFx_Series,
        samplerate: int = 22050,
        len_s: float = 5,
    ):
        super().__init__()
        self.ds_path = ds_path
        self.num_parameters = Fx.num_parameters
        self.Fx = Fx
        self.samplerate = samplerate
        self.len_s = len_s
        self.len_sp = int(len_s * samplerate)

        self.audio_path_list = []

        n = 0
        with open(os.path.join(ds_path, "audio_path_list.json"), "r") as fp:
            self.audio_path_list = json.load(fp)

    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        audio_path: str = self.audio_path_list[idx]
        full_audio_path = os.path.join(self.ds_path, audio_path)

        md = torchaudio.info(full_audio_path)

        file_length = md.num_frames

        try_again: bool = True
        while try_again:
            start_sp = np.random.randint(low=0, high=file_length - self.len_sp)
            # x, sr = torchaudio.load(
            # full_audio_path, frame_offset=start_sp, num_frames=self.len_sp
            # )

            x, sr = read_wav(
                full_audio_path,
                seek_time=start_sp / self.samplerate,
                duration=self.len_s,
            )
            x = x.mean(0)

            if x.std() > 1e-2:
                try_again = False

        x = x.numpy()

        v = np.random.rand(self.num_parameters)
        y = self.Fx(x, v)

        x = torch.Tensor(x).view(1, self.len_sp)
        y = torch.Tensor(y).view(1, self.len_sp)
        v = torch.Tensor(v)

        x = x / x.abs().amax(dim=1, keepdim=True)
        y = y / y.abs().amax(dim=1, keepdim=True)
        # print(x.std(), y.std(), v.std())
        return x, y, v


class MillionSong_2_NoFx(Dataset):
    def __init__(
        self,
        ds_path: str,
        samplerate: int = 22050,
        len_s: float = 5,
    ):
        super().__init__()
        self.ds_path = ds_path
        self.samplerate = samplerate
        self.len_s = len_s
        self.len_sp = int(len_s * samplerate)

        self.audio_path_list = []

        n = 0
        with open(os.path.join(ds_path, "audio_path_list.json"), "r") as fp:
            self.audio_path_list = json.load(fp)

    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        audio_path: str = self.audio_path_list[idx]
        full_audio_path = os.path.join(self.ds_path, audio_path)

        md = torchaudio.info(full_audio_path)

        file_length = md.num_frames

        try_again: bool = True
        while try_again:
            start_sp = np.random.randint(low=0, high=file_length - self.len_sp)
            x, sr = torchaudio.load(
                full_audio_path, frame_offset=start_sp, num_frames=self.len_sp
            )
            x = x.mean(0)

            if x.std() > 1e-2:
                try_again = False

        x = torch.Tensor(x).view(1, self.len_sp)
        return x
