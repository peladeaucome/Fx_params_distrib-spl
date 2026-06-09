import torch
import nnAudio.features
from .main import Frontend, Swish, SelfAttention, ResMultiHeadAttention
from ...utils import safe_log
from torch import Tensor
import torch.nn as nn
from typing import Callable


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
        self.append(Swish(hidden_dim))
        self.append(nn.Conv1d(hidden_dim, in_dim, kernel_size, stride, padding="same"))
        if bn:
            self.append(nn.BatchNorm1d(in_dim))

        self.out_nl = Swish(in_dim)

    def forward(self, x: Tensor):
        return self.out_nl(super().forward(x) + x)


class LayerNormSwap1d(nn.LayerNorm):
    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x = super().forward(x)
        x = torch.swapaxes(x, 1, 2)
        return x


class Conv1dNeXtBlock(nn.Sequential):
    def __init__(
        self, channels: int, kernel_size: int = 7, inv_bottleneck_factor: float = 4
    ):
        hidden_channels = int(channels * inv_bottleneck_factor)
        if kernel_size != 1:
            super().__init__(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    groups=channels,
                    dilation=1,
                ),
                LayerNormSwap1d([channels]),
                nn.Conv1d(
                    in_channels=channels, out_channels=hidden_channels, kernel_size=1
                ),
                # nn.BatchNorm1d(hidden_channels),
                Swish(hidden_channels),
                nn.Conv1d(
                    in_channels=hidden_channels, out_channels=channels, kernel_size=1
                ),
            )
        else:
            super().__init__(
                LayerNormSwap1d([channels]),
                nn.Conv1d(
                    in_channels=channels, out_channels=hidden_channels, kernel_size=1
                ),
                # nn.BatchNorm1d(hidden_channels),
                Swish(hidden_channels),
                nn.Conv1d(
                    in_channels=hidden_channels, out_channels=channels, kernel_size=1
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
        Conv1dNeXtBlock(
            channels=chan, kernel_size=kernel_size, inv_bottleneck_factor=bnf
        )
    )

    prev_chan = chan

    for chan in channels_list[1:]:
        if chan > prev_chan:
            convnext.append(
                nn.Conv1d(prev_chan, chan, kernel_size=2, stride=2, groups=prev_chan)
            )
        elif chan < prev_chan:
            convnext.append(nn.Conv1d(prev_chan, chan, kernel_size=1))

        convnext.append(
            Conv1dNeXtBlock(
                channels=chan, kernel_size=kernel_size, inv_bottleneck_factor=bnf
            )
        )
        prev_chan = chan

    if out_dim != channels_list[-1]:
        convnext.append(nn.Conv1d(channels_list[-1], out_dim, 1))
        convnext.append(Swish(out_dim))

    return convnext


class MelNeXt(Frontend):
    def __init__(
        self,
        kernel_size: int,
        channels_list: list[int],
        convnext_bottleneck_factor: int = 4,
        samplerate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        out_dim: int = 512,
    ):
        super().__init__(out_dim)
        self.n_bins = n_mels
        bnf = convnext_bottleneck_factor

        self.mel: Callable[[Tensor], Tensor] = nnAudio.features.MelSpectrogram(
            sr=samplerate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            trainable_mel=False,
            trainable_STFT=False,
        )

        self.convnext = get_ConvNeXt(
            in_dim=n_mels,
            out_dim=out_dim,
            channels_list=channels_list,
            kernel_size=kernel_size,
            bottleneck_factor=convnext_bottleneck_factor,
        )

        self.eps = nn.Parameter(torch.zeros(1, n_mels, 1))

    def forward(self, x: Tensor):
        y = x.clone()
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        y = self.mel(y)
        eps = torch.exp(self.eps)
        y = safe_log(y, eps=eps)
        y: Tensor = self.convnext(y)
        y = y.mean(2)
        return y


class MelNeXt_Attention(Frontend):
    def __init__(
        self,
        kernel_size: int,
        channels_list: list[int],
        convnext_bottleneck_factor: int = 4,
        samplerate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        out_dim: int = 512,
    ):
        super().__init__(out_dim)
        self.n_bins = n_mels
        bnf = convnext_bottleneck_factor

        self.mel: Callable[[Tensor], Tensor] = nnAudio.features.MelSpectrogram(
            sr=samplerate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            trainable_mel=False,
            trainable_STFT=False,
        )

        self.convnext = get_ConvNeXt(
            in_dim=n_mels,
            out_dim=out_dim,
            channels_list=channels_list,
            kernel_size=kernel_size,
            bottleneck_factor=convnext_bottleneck_factor,
        )

        self.eps = nn.Parameter(torch.zeros(1, n_mels, 1))
        self.attention_token = torch.nn.Parameter(torch.randn(1, channels_list[-1], 1))

        trasformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels_list[-1],
            nhead=8,
            dim_feedforward=channels_list[-1],
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            trasformer_encoder_layer, num_layers=4
        )

        self.posencoding_freqs = torch.nn.Parameter(
            torch.rand(1, channels_list[-1], 1) / 10
        )

    def forward(self, x: Tensor):
        y = x
        device = x.device
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        y = self.mel(y)
        eps = torch.exp(self.eps)
        y = safe_log(y, eps=eps)

        y: Tensor = self.convnext(y)

        bs, dim, N = y.size()
        y = torch.cat((y, self.attention_token.expand(bs, dim, 1)), dim=2)

        posenc = torch.sin(
            self.posencoding_freqs
            * (torch.arange(N + 1, device=device).unsqueeze(0).unsqueeze(0))
        )
        y=y + posenc
        y = y.swapdims(1, 2)

        y = self.transformer_encoder(y)[:, -1, :]
        return y


class MFCCNeXt(Frontend):
    def __init__(
        self,
        kernel_size: int,
        channels_list: list[int],
        convnext_bottleneck_factor: int = 4,
        samplerate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        out_dim: int = 512,
    ):
        super().__init__(out_dim)
        self.n_bins = n_mels
        bnf = convnext_bottleneck_factor

        self.mel: Callable[[Tensor], Tensor] = nnAudio.features.MelSpectrogram(
            sr=samplerate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            trainable_mel=False,
            trainable_STFT=False,
        )

        self.convnext = get_ConvNeXt(
            in_dim=n_mels,
            out_dim=out_dim,
            channels_list=channels_list,
            kernel_size=kernel_size,
            bottleneck_factor=convnext_bottleneck_factor,
        )

        self.n_mels = n_mels

    def forward(self, x: Tensor):
        y = x.clone()
        y = y / (y.std(2, keepdim=True) + 1e-3)
        y = self.mel(y)
        y = safe_log(1 + y)
        y = torch.real(torch.fft.fft(y, dim=1, n=self.n_mels, norm="ortho"))
        y: Tensor = self.convnext(y)
        y = y.mean(2)
        return y


class MelOnly(Frontend):
    def __init__(
        self,
        samplerate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__(n_mels)

        self.mel: Callable[[Tensor], Tensor] = nnAudio.features.MelSpectrogram(
            sr=samplerate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            trainable_mel=False,
            trainable_STFT=False,
        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.mel(x)
        out = safe_log(1 + x).mean(2)
        return out


class CQTNeXt(Frontend):
    def __init__(
        self,
        kernel_size: int,
        channels_list: list[int],
        convnext_bottleneck_factor: int = 4,
        samplerate: int = 22050,
        fmin: float = 32.70,
        fmax: float = None,
        n_bins=94,
        bins_per_octave: int = 12,
        hop_length: int = 512,
        out_dim: int = 512,
    ):
        super().__init__(out_dim)
        self.n_bins = n_bins
        bnf = convnext_bottleneck_factor

        self.cqt: Callable[[Tensor], Tensor] = nnAudio.features.CQT1992v2(
            sr=samplerate,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

        self.convnext = get_ConvNeXt(
            in_dim=n_bins,
            out_dim=out_dim,
            channels_list=channels_list,
            kernel_size=kernel_size,
            bottleneck_factor=convnext_bottleneck_factor,
        )

    def forward(self, x: Tensor):
        x = self.cqt(x)
        x = safe_log(1 + x)
        x = self.convnext(x)
        x = x.mean(2)
        return x


class CQT_FFT_NeXt(CQTNeXt):

    def forward(self, x: Tensor):
        x = self.cqt(x)
        x = safe_log(1 + x)
        x = torch.fft.fft(x, axis=1)
        x = torch.real(x) + torch.imag(x)
        x = self.convnext(x)
        x = x.mean(2)
        return x
