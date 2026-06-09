import torch
import nnAudio.features
from .main import Frontend
from ...utils import safe_log
from torch import Tensor
import torch.nn as nn


class TimeCQT_Encoder(Frontend):
    def __init__(
        self,
        samplerate: int = 44100,
        n_bins: int = 113,
        out_dim: int = 64,
        kernel_size: list[int] = [1, 128],
        compute_representation: bool = True,
    ):
        super().__init__(out_dim)
        self.n_bins = n_bins

        self.out_dim = out_dim
        self.compute_representation = compute_representation
        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(n_bins=n_bins, sr=samplerate)
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=out_dim,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features=out_dim),
            torch.nn.SiLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=31),
            torch.nn.BatchNorm1d(num_features=out_dim),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        if self.compute_representation:
            x = self.cqt(x)
            x = x.unsqueeze(1)
            x = safe_log(x)

        x = self.conv1(x)
        # x = torch.max(x, dim=2)
        x = x.mean(2)
        x = self.conv2(x)
        x = x.mean(2)  # Mean across time frames
        return x


class FrequencyCQT_Encoder(Frontend):
    def __init__(
        self,
        samplerate: int = 44100,
        n_bins: int = 113,
        out_dim: int = 32,
        kernel_size: list[int] = [37, 1],
        compute_representation: bool = True,
    ):
        super().__init__(out_dim=out_dim)
        self.n_bins = n_bins

        self.out_dim = out_dim
        self.compute_representation = compute_representation

        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(n_bins=n_bins, sr=samplerate)
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=out_dim,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features=out_dim),
            torch.nn.SiLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=n_bins - kernel_size[0] + 1,
            ),
            torch.nn.BatchNorm1d(num_features=out_dim),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        if self.compute_representation:
            x = self.cqt(x)
            x = x.unsqueeze(1)
            x = safe_log(x)

        x = self.conv1(x)
        x = torch.mean(x, dim=3)  # Mean across time
        x = self.conv2(x)
        x = x.squeeze(2)
        return x


class TimeFrequencyCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate: int = 44100,
        n_bins: int = 113,
        out_dim=128,
        compute_representation: bool = True,
    ):
        super(TimeFrequencyCQT_Encoder, self).__init__()
        self.n_bins = n_bins

        self.compute_representation = compute_representation

        if self.compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(n_bins=n_bins, sr=samplerate)
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.time = TimeCQT_Encoder(
            samplerate=samplerate,
            n_bins=n_bins,
            out_dim=out_dim // 2,
            compute_representation=False,
        )
        self.frequency = FrequencyCQT_Encoder(
            samplerate=samplerate,
            n_bins=n_bins,
            out_dim=out_dim // 2,
            compute_representation=False,
        )
        self.out_dim = self.time.out_dim + self.frequency.out_dim

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        x = x / (x.std(2, keepdim=True) + 1e-1)
        if self.compute_representation:
            x = self.cqt(x)
            x = x.unsqueeze(1)
            x = safe_log(x, eps=1e-3)

        out_t = self.time(x)
        out_f = self.frequency(x)

        out = torch.concat((out_t, out_f), dim=1)
        return out
