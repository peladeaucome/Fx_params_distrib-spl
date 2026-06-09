import torch
import nnAudio.features
from .main import Frontend, Swish, Swish2d
from ...utils import safe_log
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np


class Conv1dResBlock(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kernel_size: int,
        stride: int,
        nl: str = "swish",
        bn: bool = True,
    ):
        super().__init__()

        self.append(nn.Conv1d(in_dim, hidden_dim, kernel_size, stride, padding="same"))
        if bn:
            self.append(nn.BatchNorm1d(hidden_dim))
        self.append(Swish2d(hidden_dim))
        self.append(nn.Conv1d(hidden_dim, in_dim, kernel_size, stride, padding="same"))
        if bn:
            self.append(nn.BatchNorm1d(in_dim))

        self.out_nl = Swish2d(in_dim)

    def forward(self, x: Tensor):
        return self.out_nl(super().forward(x) + x)


class LayerNormSwap2d(nn.LayerNorm):
    def forward(self, x):
        x = torch.swapaxes(x, 1, 3)
        x = super().forward(x)
        x = torch.swapaxes(x, 1, 3)
        return x


class Conv2dNeXtBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int] = (1, 7),
        inv_bottleneck_factor: float = 4,
    ):
        hidden_channels = int(channels * inv_bottleneck_factor)
        if kernel_size != 1:
            super().__init__(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding="same",
                    groups=channels,
                    dilation=(1, 1),
                ),
                LayerNormSwap2d([channels]),
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 1),
                ),
                # nn.BatchNorm1d(hidden_channels),
                Swish2d(hidden_channels),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=channels,
                    kernel_size=(1, 1),
                ),
            )
        else:
            super().__init__(
                LayerNormSwap2d([channels]),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 1),
                ),
                # nn.BatchNorm1d(hidden_channels),
                Swish2d(hidden_channels),
                nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=channels,
                    kernel_size=(1, 1),
                ),
            )

    def forward(self, x):
        return super().forward(x) + x


def get_ConvNeXt(
    in_dim, out_dim, channels_list, kernel_size, bottleneck_factor
) -> nn.Sequential:
    convnext = nn.Sequential()
    chan = channels_list[0]
    bnf = bottleneck_factor

    if chan != in_dim:
        convnext.append(nn.Conv1d(in_dim, chan, kernel_size=1))

    convnext.append(
        Conv2dNeXtBlock(
            channels=chan, kernel_size=kernel_size, inv_bottleneck_factor=bnf
        )
    )

    prev_chan = chan

    for chan in channels_list[1:]:
        if chan > prev_chan:
            convnext.append(
                nn.Conv2d(
                    prev_chan, chan, kernel_size=(2, 2), stride=(2, 2), groups=prev_chan
                )
            )
        elif chan < prev_chan:
            convnext.append(nn.Conv2d(prev_chan, chan, kernel_size=1))

        convnext.append(
            Conv2dNeXtBlock(
                channels=chan, kernel_size=kernel_size, inv_bottleneck_factor=bnf
            )
        )
        prev_chan = chan

    if out_dim != channels_list[-1]:
        convnext.append(nn.Conv2d(channels_list[-1], out_dim, (1, 1)))
        convnext.append(Swish(out_dim))

    return convnext


class MeanSTFT(Frontend):
    def __init__(self, n_fft: int = 512, hop_length: int = 128):
        super().__init__(out_dim=n_fft // 2 + 1)

        self.stft = nnAudio.features.STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            output_format="Magnitude",
            verbose=False,
        )

        self.eps = nn.Parameter(torch.zeros(1, n_fft // 2 + 1))
        self.n_fft = n_fft
        self.win = torch.hann_window(n_fft).unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor):
        # x_stft: Tensor = self.stft(x)
        # out = x_stft.mean(2)
        self.win = self.win.to(x.device)
        x_, _ = torch.split(x, (self.n_fft, x.size(2) - self.n_fft), dim=2)
        out = torch.fft.rfft(x_ * self.win, dim=2, n=self.n_fft).squeeze(1).abs()

        eps = torch.exp(self.eps)
        out = safe_log(out, eps=eps)
        return out


class MeanCQT(Frontend):
    def __init__(
        self, fmin, fmax, nbins: int = 512, hop_length: int = 128, samplerate=44100
    ):
        super().__init__(out_dim=nbins)

        noctaves = np.log2(fmax / fmin)
        bins_per_octave = int(nbins / noctaves)

        self.cqt = nnAudio.features.CQT1992v2(
            sr=samplerate,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=bins_per_octave,
            filter_scale=1,
        )
        # self.stft = nnAudio.features.STFT(
        #     n_fft=n_fft, hop_length=hop_length, output_format="Magnitude"
        # )

        self.eps = nn.Parameter(torch.zeros(1, nbins))

    def forward(self, x: Tensor):
        x_cqt: Tensor = self.cqt(x)
        out = x_cqt.mean(2)

        eps = torch.exp(self.eps)
        out = safe_log(out, eps=eps)
        return out


class STFTNeXt(Frontend):
    def __init__(
        self,
        n_fft,
        hop_length,
        out_dim: int,
        kernel_size: tuple[int, int],
        channels_list: list[int],
        convnext_bottleneck_factor: int = 4,
    ):
        super().__init__(out_dim)

        self.stft = nnAudio.features.STFT(
            n_fft=n_fft, hop_length=hop_length, output_format="Magnitude"
        )

        self.eps = nn.Parameter(torch.zeros(1, n_fft // 2 + 1))

    def forward(self, x: Tensor):
        y = x.clone()

        y = self.stft(y)
        eps = torch.exp(self.eps)
        y = safe_log(y, eps=eps)
