import torch
import nnAudio.features
import torchmetrics
import torchaudio
from .utils import safe_log
from .ddafx.filters.utils import fftfilt
import torch.nn as nn
from torch import Tensor
from typing import Literal, Iterable, Optional, Union
import numpy as np
from torch import Tensor
from typing import List
from .peaq_numpy import PEAQ


def l2norm(input: Tensor, target: Tensor = 0):
    return torch.sum(torch.square(input - target), dim=-1)


def si_sdr(input: Tensor, target: Tensor):
    batch_size, num_channels, num_samples = input.size()

    coeff = torch.sum(input * target, dim=-1) / l2norm(target)
    coeff = coeff.reshape(batch_size, num_channels, 1)
    out = l2norm(coeff * target) / l2norm(coeff * target, input)
    out = 10 * torch.log10(out)
    return out.mean(1)


class SISDR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return si_sdr(x, y)


class Neg_SISDR(SISDR):
    def forward(self, x, y):
        return -super().forward(x, y)


def abs_params_loss(input: torch.Tensor, target: torch.Tensor):
    v = input * 2 - 1
    vhat = target * 2 - 1

    return (v.abs() - vhat.abs()).square().mean()


class MR_STFT_Revisited(torch.nn.Module):
    def __init__(
        self,
        n_ffts: list = [67, 127, 257, 509, 1021, 2053],
        hop_lengths: list = None,
        window_sizes: list = None,
        samplerate: int = 44100,
        window: Literal["hann", "hamming", "flattop"] = "flattop",
    ):
        super().__init__()
        self.stft_list = torch.nn.ModuleList()

        if hop_lengths is None:
            hop_lengths = [N // 2 for N in n_ffts]
        if window_sizes is None:
            window_sizes = n_ffts

        for i, n_fft in enumerate(n_ffts):
            hop_length = hop_lengths[i]
            window_size = window_sizes[i]
            self.stft_list.append(
                nnAudio.features.STFT(
                    n_fft=n_fft,
                    win_length=window_size,
                    hop_length=hop_length,
                    sr=samplerate,
                    trainable=False,
                    output_format="Magnitude",
                    verbose=False,
                    freq_scale="no",
                    window=window,
                )
            )

        self.l2 = torch.nn.MSELoss(reduction="none")

    def transform(self, x: Tensor):
        return safe_log(x.abs() + 1)

    def forward(self, input, target):
        out = 0
        for stft in self.stft_list:
            input_stft: Tensor = self.transform(stft(input))
            target_stft: Tensor = self.transform(stft(target))

            out = out + (input_stft - target_stft).square().mean(dim=(1, 2))

        out = out / len(self.stft_list)

        return out * 20


class MR_STFT_Revisited_Norm(MR_STFT_Revisited):
    def forward(self, input, target):
        return super().forward(rms_norm(input), rms_norm(target))


class MR_STFT_Loss(torch.nn.Module):
    def __init__(
        self,
        n_ffts: Iterable[int] = [256, 1024, 4096],
        hop_lengths: Iterable[int] = [128, 512, 2048],
        window_sizes: Iterable[int] = [256, 1024, 4096],
        samplerate: int = 44100,
        eps: float = 0.1,
    ):
        super(MR_STFT_Loss, self).__init__()
        self.stft_list = torch.nn.ModuleList()

        for i, n_fft in enumerate(n_ffts):
            hop_length = hop_lengths[i]
            window_size = window_sizes[i]
            self.stft_list.append(
                nnAudio.features.STFT(
                    n_fft=n_fft,
                    win_length=window_size,
                    hop_length=hop_length,
                    sr=samplerate,
                    trainable=False,
                    output_format="Magnitude",
                    verbose=False,
                    freq_scale="no",
                )
            )

        self.l2 = torch.nn.MSELoss(reduction="none")

        self.eps = eps

    def transform(self, x):
        return safe_log(x)

    @staticmethod
    def frob_norm(X: Tensor):
        return X.square().mean(dim=(1, 2))

    def forward(self, input: Tensor, target: Tensor):
        input = rms_norm(input)
        target = rms_norm(target)
        sc = 0
        sm = 0
        for stft in self.stft_list:
            input_stft: Tensor = stft(input).abs()
            target_stft: Tensor = stft(target).abs()

            input_stft_log = safe_log(input_stft, eps=self.eps)
            target_stft_log = safe_log(target_stft, eps=self.eps)

            sm = sm + (input_stft_log - target_stft_log).abs().mean(dim=(1, 2))
            sc = sc + self.frob_norm(target_stft - input_stft) / (
                self.frob_norm(target_stft) + 1e-6
            )

        out = (sm + sc) / len(self.stft_list)

        return out


MR_STFT_LogMagMSE = MR_STFT_Loss


class LogMel_Loss(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        win_length: int = None,
        n_mels: int = 128,
        hop_length: int = 512,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        htk: bool = False,
        fmin: float = 0.0,
        fmax: float = None,
        norm: float = 1,
        verbose: bool = False,
        eps: float = 1,
    ):
        super().__init__()

        self.mel = nnAudio.features.MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            htk=htk,
            fmin=fmin,
            fmax=fmax,
            norm=norm,
            verbose=verbose,
        )
        self.eps = eps

    def forward(self, input_wav: Tensor, target_wav: Tensor):
        input_mel: Tensor = safe_log(self.mel(input_wav), eps=self.eps)
        target_mel: Tensor = safe_log(self.mel(target_wav), eps=self.eps)

        return (input_mel - target_mel).square().mean(dim=(1, 2))


class NormalizedLogMel_Loss(LogMel_Loss):
    def forward(self, input: Tensor, target: Tensor):
        input = rms_norm(input, 1e-2)
        target = rms_norm(target, 1e-2)
        return super().forward(input, target)


def rms_norm(x: torch.Tensor, eps: float = 1e-6):
    x = (x - torch.mean(x, dim=2, keepdim=True)) / (
        torch.std(x, dim=2, keepdim=True) + eps
    )
    return x


def absmse(input: Tensor, target: Tensor):
    return torch.nn.functional.mse_loss(input.abs(), target.abs())


def pimse(input: Tensor, target: Tensor):
    out1 = (input - target).square().mean(dim=(1, 2))
    out2 = (input + target).square().mean(dim=(1, 2))
    return torch.minimum(out1, out2)


def neg_sisdr(input, target):
    return -si_sdr(input, target)


def dct(x: Tensor, **fft_kwargs):
    out = torch.fft.rfft(x, **fft_kwargs)
    return torch.real(out)


class MFCC_Loss(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        win_length: int = None,
        hop_length: int = 512,
        n_mels: int = 128,
        num_features=32,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        htk: bool = False,
        fmin: float = 0.0,
        fmax: float = None,
        norm: float = 1,
        verbose: bool = False,
        eps: float = 0.01,
    ):
        super().__init__()

        self.mel = nnAudio.features.MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            htk=htk,
            fmin=fmin,
            fmax=fmax,
            norm=norm,
            verbose=verbose,
        )
        self.num_features = num_features
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor):
        # input = rms_norm(input.clone())
        # target = rms_norm(target.clone())

        input_mfcc: Tensor = dct(
            safe_log(self.mel(input), eps=self.eps),
            n=self.num_features,
            dim=1,
            norm="ortho",
        )
        target_mfcc: Tensor = dct(
            safe_log(self.mel(target), eps=self.eps),
            n=self.num_features,
            dim=1,
            norm="ortho",
        )
        # input_mel: Tensor = self.mel(input)
        # target_mel: Tensor = self.mel(target)

        return (input_mfcc - target_mfcc).square().mean(dim=(1, 2))


class SpectralOT_Loss(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        win_length: int = None,
        hop_length: int = 512,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        fmin: float = 0.0,
        fmax: float = None,
        verbose: bool = False,
        eps: float = 0.001,
        weight_loud=1.0,
    ):
        super().__init__()

        self.stft = nnAudio.features.STFT(
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            fmin=fmin,
            fmax=fmax,
            verbose=verbose,
            output_format="Magnitude",
        )
        # self.stft = nnAudio.features.CQT1992v2(
        #     sr=sr,
        #     hop_length=hop_length,
        #     fmin=fmin,
        #     fmax=fmax,
        #     n_bins=n_fft//2,
        #     filter_scale=0.5,
        #     output_format="Magnitude"
        # )
        self.n_fft = n_fft

        self.weight_loud = weight_loud
        self.eps = eps

    def forward(self, y: Tensor, x: Tensor):
        bs = x.size(0)
        fx = self.stft(x)
        fy = self.stft(y)

        nbins = fx.size(1)

        Fx = torch.cumsum(fx, dim=1)
        # fx = fx / (Fx[:, -1, :].unsqueeze(1))
        Fx = Fx / (Fx[:, -1, :].unsqueeze(1))

        Fy = torch.cumsum(fy, dim=1)
        # fy = fy / (Fy[:, -1, :].unsqueeze(1))
        Fy = Fy / (Fy[:, -1, :].unsqueeze(1))

        f: torch.Tensor = torch.fft.rfftfreq(self.n_fft).to(x.device).view(1, 1, -1)
        f = f.expand(bs, fy.size(2), -1).contiguous()

        quantiles = torch.sort(torch.cat((Fx, Fy), dim=1), dim=1)[0]
        quantiles = quantiles.contiguous()

        Fx = Fx.swapdims(1, 2).contiguous()
        idx_x = torch.searchsorted(Fx, quantiles.swapdims(1, 2))
        idx_x = idx_x.clip(0, nbins - 1).contiguous()

        Finv_x = torch.gather(f, index=idx_x, dim=2).swapdims(1, 2)

        Fy = Fy.swapdims(1, 2).contiguous()
        idx_y = torch.searchsorted(Fy, quantiles.swapdims(1, 2))
        idx_y = idx_y.clip(0, nbins - 1).contiguous()

        Finv_y = torch.gather(f, index=idx_y, dim=2).swapdims(1, 2)

        dr = quantiles[:, 1:, :] - quantiles[:, :-1, :]
        OT_loss = ((Finv_y - Finv_x)[:, 1:, :].square() * dr).sum(1).mean(1) * 100

        return OT_loss.clone()


class SpectralOT_Log_Loss(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        nbins: int = 512,
        win_length: int = None,
        hop_length: int = 512,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        fmin: float = 20,
        fmax: float = 12000,
        verbose: bool = False,
        eps: float = 0.001,
        weight_loud=1.0,
    ):
        super().__init__()

        fmin = fmin
        fmax = fmax
        num_octaves = np.log2(fmax / fmin)
        bins_per_octave = int(nbins / num_octaves)
        self.cqt = nnAudio.features.CQT1992v2(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=bins_per_octave,
            filter_scale=1,
            output_format="Magnitude",
        )

        self.weight_loud = weight_loud
        self.eps = eps
        self.sr=sr

    def forward(self, y: Tensor, x: Tensor):
        bs = x.size(0)
        fx = safe_log(self.cqt(x), eps=1)
        fy = safe_log(self.cqt(y), eps=1)

        nbins = fx.size(1)

        Fx = torch.cumsum(fx, dim=1)
        # fx = fx / (Fx[:, -1, :].unsqueeze(1))
        Fx = Fx / (Fx[:, -1, :].unsqueeze(1)).contiguous()

        Fy = torch.cumsum(fy, dim=1)
        # fy = fy / (Fy[:, -1, :].unsqueeze(1))
        Fy = Fy / (Fy[:, -1, :].unsqueeze(1)).contiguous()

        f = torch.tensor(self.cqt.frequencies, device=x.device, dtype=x.dtype).view(1, 1, -1)/self.sr
        # f: torch.Tensor = torch.fft.rfftfreq(self.n_fft).to(x.device).view(1, 1, -1)
        f = f.expand(bs, fy.size(2), -1).contiguous()


        quantiles = torch.sort(torch.cat((Fx, Fy), dim=1), dim=1)[0]
        quantiles = quantiles.contiguous()

        Fx = Fx.swapdims(1, 2).contiguous()
        idx_x = torch.searchsorted(Fx, quantiles.swapdims(1, 2))
        idx_x = idx_x.clip(0, nbins - 1).contiguous()

        Finv_x = torch.gather(f, index=idx_x, dim=2).swapdims(1, 2)

        Fy = Fy.swapdims(1, 2).contiguous()
        idx_y = torch.searchsorted(Fy, quantiles.swapdims(1, 2))
        idx_y = idx_y.clip(0, nbins - 1).contiguous()

        Finv_y = torch.gather(f, index=idx_y, dim=2).swapdims(1, 2)

        dr = quantiles[:, 1:, :] - quantiles[:, :-1, :]
        OT_loss = ((Finv_y - Finv_x)[:, 1:, :].square() * dr).sum(1).mean(1) * 100

        return OT_loss


class SpectralOT_RMS_Loss(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        win_length: int = None,
        hop_length: int = 512,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        fmin: float = 0.0,
        fmax: float = None,
        verbose: bool = False,
        eps: float = 0.001,
        weight_loud=1.0,
    ):
        super().__init__()

        self.stft = nnAudio.features.STFT(
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            fmin=fmin,
            fmax=fmax,
            verbose=verbose,
            output_format="Magnitude",
        )
        self.n_fft = n_fft

        self.weight_loud = weight_loud
        self.eps = eps

    def rms(self, x: Tensor):
        out = x.square().mean((1, 2)).sqrt()
        return out

    def forward(self, y: Tensor, x: Tensor):
        bs = x.size(0)
        fx = self.stft(x)
        fy = self.stft(y)

        nbins = fx.size(1)

        Fx = torch.cumsum(fx, dim=1)
        fx = fx / (Fx[:, -1, :].unsqueeze(1))
        Fx = Fx / (Fx[:, -1, :].unsqueeze(1)).contiguous()

        Fy = torch.cumsum(fy, dim=1)
        fy = fy / (Fy[:, -1, :].unsqueeze(1))
        Fy = Fy / (Fy[:, -1, :].unsqueeze(1)).contiguous()

        f: torch.Tensor = torch.fft.rfftfreq(self.n_fft).to(x.device).view(1, 1, -1)
        f = f.expand(bs, fy.size(2), -1).contiguous()

        quantiles = torch.sort(torch.cat((Fx, Fy), dim=1), dim=1)[0]
        quantiles = quantiles.contiguous()

        Fx = Fx.swapdims(1, 2).contiguous()
        idx_x = torch.searchsorted(Fx, quantiles.swapdims(1, 2))
        idx_x = idx_x.clip(0, nbins - 1).contiguous()

        Finv_x = torch.gather(f, index=idx_x, dim=2).swapdims(1, 2)

        Fy = Fy.swapdims(1, 2).contiguous()
        idx_y = torch.searchsorted(Fy, quantiles.swapdims(1, 2))
        idx_y = idx_y.clip(0, nbins - 1).contiguous()

        Finv_y = torch.gather(f, index=idx_y, dim=2).swapdims(1, 2)

        dr = quantiles[:, 1:, :] - quantiles[:, :-1, :]
        OT_loss = ((Finv_y - Finv_x)[:, 1:, :].square() * dr).sum(1).mean(1) * 100

        RMS_loss = (safe_log(self.rms(x)) - safe_log(self.rms(y))).square()

        return OT_loss + RMS_loss * self.weight_loud


class LDRLoss(nn.Module):
    def __init__(self, long_t: float, short_t: float, samplerate: int):
        super().__init__()
        short_alpha = 1 - np.exp(-1 / (short_t * samplerate))
        long_alpha = 1 - np.exp(-1 / (long_t * samplerate))

        self.long_a = torch.Tensor([1, long_alpha - 1]).unsqueeze(0)
        self.long_b = torch.Tensor([long_alpha, 1]).unsqueeze(0)
        self.long_pad = int(long_t * samplerate)

        self.short_a = torch.Tensor([1, short_alpha - 1]).unsqueeze(0)
        self.short_b = torch.Tensor([short_alpha, 1]).unsqueeze(0)
        self.short_pad = int(short_t * samplerate)

        self.align_shift = int((long_t - short_t) * samplerate * 0.0005)

    def roll(self, x: Tensor):
        bs, c, N = x.size()
        _, x = torch.split(x, (self.align_shift, N - self.align_shift), dim=2)
        x = torch.cat((x, torch.zeros(bs, c, self.align_shift, device=x.device)), dim=2)
        return x

    def get_RMS(self, x: Tensor, a: Tensor, b: Tensor, pad):
        x2 = x.square()
        env = fftfilt(x2.squeeze(1), a, b, pad=pad).unsqueeze(1)
        return env

    def get_diff_env(self, x: Tensor):
        bs = x.size(0)
        env_short = safe_log(
            self.get_RMS(
                x,
                self.short_a.expand(bs, 2).to(x.device),
                self.short_b.expand(bs, 2).to(x.device),
                self.short_pad,
            )
        )
        env_long = safe_log(
            self.get_RMS(
                self.roll(x),
                self.long_a.expand(bs, 2).to(x.device),
                self.long_b.expand(bs, 2).to(x.device),
                self.long_pad,
            )
        )

        return env_short - env_long

    def forward(self, x, y):
        return (self.get_diff_env(x) - self.get_diff_env(y)).abs().mean(2).mean(1)

    def to(self, device: torch.device):
        self.long_a = self.long_a.to(device)
        self.long_b = self.long_b.to(device)
        self.short_a = self.short_a.to(device)
        self.short_b = self.short_b.to(device)
        self.device = device
        return self

    def apply(self, fn: callable):
        fn(self)
        fn(self.long_a)
        fn(self.long_b)
        fn(self.short_a)
        fn(self.short_b)
        return self

    def cpu(self):
        self.long_a = self.long_a.cpu()
        self.long_b = self.long_b.cpu()
        self.short_a = self.short_a.cpu()
        self.short_b = self.short_b.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.long_a = self.long_a.cuda()
        self.long_b = self.long_b.cuda()
        self.short_a = self.short_a.cuda()
        self.short_b = self.short_b.cuda()
        return self


class two_f_Model(nn.Module):
    def __init__(self, samplerate):
        super().__init__()
        self.samplerate = samplerate
        self.peaq = PEAQ(mode="basic", Amax=1, verbose=False)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.samplerate, new_freq=48000
        )

    def forward(self, x: Tensor, y: Tensor):
        device=x.device
        with torch.no_grad():
            self.resampler = self.resampler.to(device)
            if self.samplerate != 48000:
                x = self.resampler(x)
                y = self.resampler(y)

            bs, _, N = x.size()

            x = x / x.std(dim=2, keepdim=True)
            y = y / y.std(dim=2, keepdim=True)

            x_max = x.abs().amax(dim=2, keepdim=True)
            y_max = y.abs().amax(dim=2, keepdim=True)

            M = torch.maximum(x_max, y_max)

            x = x / M
            y = y / M

            x = x.expand(bs, 2, N)
            y = y.expand(bs, 2, N)

            x = x.cpu().numpy()
            y = y.cpu().numpy()

            out = torch.zeros(bs, device=device)
            for batch_idx in range(bs):
                MMS = self.peaq.compute_2fmodel_from_waveform(
                    y[batch_idx], x[batch_idx]
                )
                out[batch_idx] = MMS
        return out.to(device)


class SumLosses(nn.Module):
    def __init__(self, weights: List[float] = [], loss_fns: List[nn.Module] = []):
        super().__init__()

        self.losses = nn.ModuleList(loss_fns)
        self.weights = weights

    def forward(self, pred, target):
        losses = [fn(pred, target) for fn in self.losses]
        return sum(map(lambda w, x: w * x, self.weights, losses))
