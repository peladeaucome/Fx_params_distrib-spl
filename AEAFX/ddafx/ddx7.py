from .main import DDAFX
import torch
import numpy as np
from torch import Tensor
from typing import Literal, Union, Optional


class DDX7_algo1(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[min_ratio, max_ratio] for i in range(5)]
        rp = rp + [[0, mod_max] for i in range(5)]

        self.base_freq = base_freq
        self.min_ratio = torch.Tensor(min_ratio * torch.ones(1, 5))
        self.max_ratio = torch.Tensor(max_ratio * torch.ones(1, 5))
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm
        super().__init__(num_parameters=10, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, freq_ratios: Tensor, amps: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        freqs = freqs.unsqueeze(2)
        amps = amps.unsqueeze(2)

        op1 = torch.sin(t * freqs[:, 0] * 2 * np.pi) * amps[:, 0]
        op2 = torch.sin(t * freqs[:, 1] * 2 * np.pi + op1) * amps[:, 1]
        op3 = torch.sin(t * freqs[:, 2] * 2 * np.pi + op2) * amps[:, 2]

        op4 = torch.sin(t * freqs[:, 3] * 2 * np.pi) * amps[:, 3]
        op5 = torch.sin(t * freqs[:, 4] * 2 * np.pi + op4) * amps[:, 4]

        out = op3 + op5
        return torch.Tensor(out.unsqueeze(1))

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w):
        w_f, w_a = torch.split(w, 5, dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = min_ratio.log()
        log_max_r = max_ratio.log()

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps = torch.zeros_like(w_a)

        amps[:, 0:2] = w_a[:, 0:2] * self.mod_max
        amps[:, 3] = w_a[:, 3] * self.mod_max

        amps[:, 2] = w_a[:, 2]
        amps[:, 4] = w_a[:, 4]
        return amps, freq_ratios

    def to(self, device: torch.device):
        self.min_ratio = self.min_ratio.to(device)
        self.max_ratio = self.max_ratio.to(device)
        super().to(device)
        return self

    def apply(self, fn: callable):
        fn(self.min_ratio)
        fn(self.max_ratio)
        super().apply(fn)
        return self

    def cpu(self):
        self.min_ratio = self.min_ratio.cpu()
        self.max_ratio = self.max_ratio.cpu()
        super().cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.min_ratio = self.min_ratio.cuda(device)
        self.max_ratio = self.max_ratio.cuda(device)
        super().cuda(device)
        return self


class DDX7_algo56(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[min_ratio, max_ratio] for i in range(6)]
        rp = rp + [[0, mod_max] for i in range(6)]

        self.base_freq = base_freq
        self.min_ratio = torch.Tensor(min_ratio * torch.ones(1, 6))
        self.max_ratio = torch.Tensor(max_ratio * torch.ones(1, 6))
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm
        super().__init__(num_parameters=12, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, amps: Tensor, freq_ratios: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        freqs = freqs.unsqueeze(2)
        amps = amps.unsqueeze(2)

        op2 = torch.cos(t * freqs[:, 1] * 2 * np.pi) * amps[:, 1]
        op1 = torch.cos(t * freqs[:, 0] * 2 * np.pi + op2) * amps[:, 0]

        op4 = torch.cos(t * freqs[:, 3] * 2 * np.pi) * amps[:, 3]
        op3 = torch.cos(t * freqs[:, 2] * 2 * np.pi + op4) * amps[:, 2]

        op6 = torch.cos(t * freqs[:, 5] * 2 * np.pi) * amps[:, 5]
        op5 = torch.cos(t * freqs[:, 4] * 2 * np.pi + op6) * amps[:, 4]

        out = op1 + op3 + op5
        return out.unsqueeze(1)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, (6, 3), dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = min_ratio.log()
        log_max_r = max_ratio.log()

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps_p = w_a * 99
        amps_db = amps_p * 3 / 4 - 99 * 3 / 4

        amps = torch.pow(10, amps_db / 20)

        amps_out = torch.zeros_like(amps)
        amps_out[:, 0::2] = amps[:, 0::2]
        amps_out[:, 1::2] = amps[:, 1::2] * self.mod_max
        # amps[:,1] = w_a[:,1]*self.mod_max
        # amps[:,3] = w_a[:,3]*self.mod_max
        # amps[:,5] = w_a[:,5]*self.mod_max
        # amps[:,0] = w_a[:,0]
        # amps[:,2] = w_a[:,2]
        # amps[:,4] = w_a[:,4]
        return amps_out, freq_ratios

        # return amps, freq_ratios

    def to(self, device: torch.device):
        self.min_ratio = self.min_ratio.to(device)
        self.max_ratio = self.max_ratio.to(device)
        super().to(device)
        return self

    def apply(self, fn: callable):
        fn(self.min_ratio)
        fn(self.max_ratio)
        super().apply(fn)
        return self

    def cpu(self):
        self.min_ratio = self.min_ratio.cpu()
        self.max_ratio = self.max_ratio.cpu()
        super().cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.min_ratio = self.min_ratio.cuda(device)
        self.max_ratio = self.max_ratio.cuda(device)
        super().cuda(device)
        return self


class DDX7_algo56_simple(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[0, mod_max] for i in range(6)]
        rp = rp + [[min_ratio, max_ratio] for i in range(3)]

        self.base_freq = base_freq
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm

        self.params_names = [
            "Amp. 1",
            "Amp. 2",
            "Amp. 3",
            "Amp. 4",
            "Amp. 5",
            "Amp. 6",
            "Freq. 2",
            "Freq. 4",
            "Freq. 6",
        ]
        super().__init__(num_parameters=9, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, amps: Tensor, freq_ratios: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        # freqs = freqs.unsqueeze(2)
        # amps = amps.unsqueeze(2)

        f2, f4, f6 = torch.split(freqs, [1, 1, 1], dim=1)
        a1, a2, a3, a4, a5, a6 = torch.split(amps, [1, 1, 1, 1, 1, 1], dim=1)

        op2 = torch.sin(t * f2 * 2 * np.pi) * a2
        op1 = torch.sin(t * 2 * np.pi * self.base_freq + op2) * a1

        op4 = torch.sin(t * f4 * 2 * np.pi) * a4
        op3 = torch.sin(t * 2 * np.pi * self.base_freq + op4) * a3

        op6 = torch.sin(t * f6 * 2 * np.pi) * a6
        op5 = torch.sin(t * 2 * np.pi * self.base_freq + op6) * a5

        out = op1 + op3 + op5
        return out.unsqueeze(1)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, [6, 3], dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = np.log(min_ratio)
        log_max_r = np.log(max_ratio)

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps_p = w_a * 99
        amps_db = amps_p * 3 / 4 - 99 * 3 / 4

        amps = torch.pow(10, amps_db / 20)

        amps_out = torch.zeros_like(amps)
        amps_out[:, 0::2] = amps[:, 0::2]
        amps_out[:, 1::2] = amps[:, 1::2] * self.mod_max
        # amps[:,1] = w_a[:,1]*self.mod_max
        # amps[:,3] = w_a[:,3]*self.mod_max
        # amps[:,5] = w_a[:,5]*self.mod_max
        # amps[:,0] = w_a[:,0]
        # amps[:,2] = w_a[:,2]
        # amps[:,4] = w_a[:,4]
        return amps_out, freq_ratios

        # return amps, freq_ratios


class DDX7_algo34_simple(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[0, mod_max] for i in range(6)]
        rp = rp + [[min_ratio, max_ratio] for i in range(4)]

        self.base_freq = base_freq
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm
        super().__init__(num_parameters=9, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, amps: Tensor, freq_ratios: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        freqs = freqs.unsqueeze(2)
        amps = amps.unsqueeze(2)

        op3 = torch.sin(t * 2 * np.pi * self.base_freq * freqs[:, 1]) * amps[:, 2]
        op2 = torch.sin(t * freqs[:, 0] * 2 * np.pi) * amps[:, 1]
        op1 = torch.sin(t * 2 * np.pi * self.base_freq + op2) * amps[:, 0]

        op4 = torch.sin(t * freqs[:, 1] * 2 * np.pi) * amps[:, 3]

        op6 = torch.sin(t * freqs[:, 2] * 2 * np.pi) * amps[:, 5]
        op5 = torch.sin(t * 2 * np.pi * self.base_freq + op6) * amps[:, 4]

        out = op1 + op3 + op5
        return out.unsqueeze(1)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, [6, 3], dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = np.log(min_ratio)
        log_max_r = np.log(max_ratio)

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps_p = w_a * 99
        amps_db = amps_p * 3 / 4 - 99 * 3 / 4

        amps = torch.pow(10, amps_db / 20)

        amps_out = torch.zeros_like(amps)
        amps_out[:, 0::2] = amps[:, 0::2]
        amps_out[:, 1::2] = amps[:, 1::2] * self.mod_max
        # amps[:,1] = w_a[:,1]*self.mod_max
        # amps[:,3] = w_a[:,3]*self.mod_max
        # amps[:,5] = w_a[:,5]*self.mod_max
        # amps[:,0] = w_a[:,0]
        # amps[:,2] = w_a[:,2]
        # amps[:,4] = w_a[:,4]
        return amps_out, freq_ratios


class DDX7_1_osc(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[0, mod_max] for i in range(1)]
        rp = rp + [[min_ratio, max_ratio] for i in range(1)]

        self.base_freq = base_freq
        self.min_ratio = torch.tensor(torch.ones(1, 1) * min_ratio, dtype=torch.float32)
        self.max_ratio = torch.tensor(torch.ones(1, 1) * max_ratio, dtype=torch.float32)
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm
        super().__init__(num_parameters=2, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, amps: Tensor, freq_ratios: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        # freqs = freqs.unsqueeze(2)
        # amps = amps.unsqueeze(2)

        op2 = torch.sin(t * freqs * 2 * np.pi) * amps
        op1 = torch.sin(t * 2 * np.pi * self.base_freq + op2)

        return op1.unsqueeze(1)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, [1, 1], dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = min_ratio.log()
        log_max_r = max_ratio.log()

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps_p = w_a * 99
        amps_db = amps_p * 3 / 4 - 99 * 3 / 4

        amps = torch.pow(10, amps_db / 20)

        amps = amps * self.mod_max
        return amps, freq_ratios

        # return amps, freq_ratios

    def to(self, device: torch.device):
        super().to(device)
        self.min_ratio = self.min_ratio.to(device)
        self.max_ratio = self.max_ratio.to(device)
        return self

    def apply(self, fn: callable):
        super().apply(fn)
        fn(self.min_ratio)
        fn(self.max_ratio)
        return self

    def cpu(self):
        super().cpu()
        self.min_ratio = self.min_ratio.cpu()
        self.max_ratio = self.max_ratio.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        self.min_ratio = self.min_ratio.cuda(device)
        self.max_ratio = self.max_ratio.cuda(device)
        return self


class DDX7_3_osc(DDAFX):
    def __init__(
        self,
        base_freq: float = 440,
        mod_max: float = 2,
        min_ratio=0.1,
        max_ratio=10,
        samplerate: int = 44100,
        ratio_norm: Literal["lin", "log"] = "lin",
    ):
        rp = [[0, mod_max] for i in range(2)]
        rp = rp + [[min_ratio, max_ratio] for i in range(2)]

        self.base_freq = base_freq
        self.min_ratio = torch.tensor(torch.ones(1, 1) * min_ratio, dtype=torch.float32)
        self.max_ratio = torch.tensor(torch.ones(1, 1) * max_ratio, dtype=torch.float32)
        self.mod_max = mod_max
        self.ratio_norm = ratio_norm
        super().__init__(num_parameters=4, ranges_parameters=rp, samplerate=samplerate)

    def process(self, x: Tensor, amps: Tensor, freq_ratios: Tensor):
        batchSize, _, N = x.size()
        device = x.device
        sr = self.samplerate

        t = torch.arange(N, device=device).unsqueeze(0) / sr

        freqs = freq_ratios * self.base_freq

        amp3, amp2 = torch.split(amps, [1, 1], dim=1)
        freq3, freq2 = torch.split(freqs, [1, 1], dim=1)

        # freqs = freqs.unsqueeze(2)
        # amps = amps.unsqueeze(2)
        op3 = torch.sin(t * freq3 * 2 * np.pi * self.base_freq) * amp3
        op2 = torch.sin(t * freq2 * 2 * np.pi * self.base_freq + op3) * amp2
        op1 = torch.sin(t * 2 * np.pi * self.base_freq + op2)

        return op1.unsqueeze(1)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        amps, freq_ratios = self.denormalize_parameters(w)
        return self.process(x, amps, freq_ratios)

    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, [2, 2], dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = min_ratio.log()
        log_max_r = max_ratio.log()

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps_p = w_a * 99
        amps_db = amps_p * 3 / 4 - 99 * 3 / 4

        amps = torch.pow(10, amps_db / 20)

        amps = amps * self.mod_max
        return amps, freq_ratios

        # return amps, freq_ratios

    def to(self, device: torch.device):
        super().to(device)
        self.min_ratio = self.min_ratio.to(device)
        self.max_ratio = self.max_ratio.to(device)
        return self

    def apply(self, fn: callable):
        super().apply(fn)
        fn(self.min_ratio)
        fn(self.max_ratio)
        return self

    def cpu(self):
        super().cpu()
        self.min_ratio = self.min_ratio.cpu()
        self.max_ratio = self.max_ratio.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        self.min_ratio = self.min_ratio.cuda(device)
        self.max_ratio = self.max_ratio.cuda(device)
        return self


class DDX7_1_osc_lin(DDX7_1_osc):
    def denormalize_parameters(self, w: Tensor):
        w_a, w_f = torch.split(w, [1, 1], dim=1)

        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        log_min_r = min_ratio.log()
        log_max_r = max_ratio.log()

        if self.mode == "analysis":
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()
        else:
            if self.ratio_norm == "lin":
                freq_ratios = w_f * (max_ratio - min_ratio) + min_ratio
            else:
                freq_ratios = (w_f * (log_max_r - log_min_r) + log_min_r).exp()

        amps = w_a * self.mod_max
        return amps, freq_ratios

        # return amps, freq_ratios
