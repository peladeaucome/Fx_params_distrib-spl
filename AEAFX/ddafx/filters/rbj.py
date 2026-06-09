import torch
from torch import Tensor
from torch.nn import Module
from ..main import DDAFX, DDAFXChain
import numpy as np
import torchaudio
from .utils import fftfilt, chain_fftfilt
from typing import Union, Optional
import nnAudio.features


class Band(DDAFX):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        if ranges_parameters is None:
            ranges_parameters = [[20, 20000], [-10, 10], [0.1, 3]]
        super().__init__(
            num_parameters=3,
            ranges_parameters=ranges_parameters,
            samplerate=samplerate,
        )

        self.name = "Band"
        self.params_names = ["Frequency", "Gain", "Q"]
        # self.stft:Module = nnAudio.features.STFT(
        #     n_fft=8192,
        #     win_length=4096,
        #     hop_length=2048,
        #     verbose=False,
        #     window='hann',
        #     iSTFT=True,
        # )

    def denormalize_parameters(self, w: Tensor):
        v = torch.zeros_like(w)

        m_lin = self.ranges_parameters[:, 0].reshape(1, 3)
        M_lin = self.ranges_parameters[:, 1].reshape(1, 3)

        m_log = torch.log(m_lin)
        M_log = torch.log(M_lin)

        v[:, 0] = torch.exp(w[:, 0] * (M_log[:, 0] - m_log[:, 0]) + m_log[:, 0])
        v[:, 1] = w[:, 1] * (M_lin[:, 1] - m_lin[:, 1]) + m_lin[:, 1]
        v[:, 2] = w[:, 2] * (M_lin[:, 2] - m_lin[:, 2]) + m_lin[:, 2]
        # v[:, 2] = torch.exp(w[:, 2] * (M_log[:, 2] - m_log[:, 2]) + m_log[:, 2])

        # m = torch.log(self.ranges_parameters[0:3:2, 0]).unsqueeze(0)
        # M = torch.log(self.ranges_parameters[0:3:2, 1]).unsqueeze(0)
        # v[:, 0:3:2] = torch.exp(w[:, 0:3:2] * (M - m) + m)
        return v

    def process(self, x: Tensor, v: Tensor):
        b, a = self.get_coeffs(v)
        device = v.device
        N = x.size(2)

        # y = torchaudio.functional.lfilter(
        #     x, a_coeffs=a, b_coeffs=b, clamp=False, batching=True
        # )
        x = x.squeeze(1)

        y = fftfilt(x, a_coeffs=a, b_coeffs=b)
        y = y.unsqueeze(1)

        # # self.stft=self.stft.to(device)
        # # self.istft=self.istft.to(device)
        # # print("1")
        # n_fft = 8192
        # # print("2")
        # x_stft:Tensor = self.stft(x)
        # # print("3")
        # a_fft = torch.fft.rfft(a, n=n_fft, dim=1)
        # # print("4")
        # b_fft = torch.fft.rfft(b, n=n_fft, dim=1)
        # # print("5")
        # h_fft: Tensor = b_fft / a_fft
        # # print("6")
        # h_fft = h_fft.unsqueeze(2).unsqueeze(3)

        # x_real, x_imag = x_stft.split([1, 1],dim=3)
        # # print(type(x_real))
        # y_real = x_real*torch.real(h_fft)
        # # print(type(x_real))
        # y_imag = x_imag*torch.imag(h_fft)
        # y_stft = torch.cat((y_real, y_imag), dim=3)

        # # print("7")
        # # y_stft = x_stft * h_fft
        # # print(x_stft.size(), y_stft.size())
        # y = self.stft.inverse(y_stft, length=N).unsqueeze(1)
        # # print(y.size())
        return y

    def get_coeffs(self, v: Tensor) -> Tensor:
        raise NotImplementedError("please implement a get_coeffs method")

    # def to(self, device: torch.device):
    #     # print('to', device)
    #     self.ranges_parameters = self.ranges_parameters.to(device)
    #     self.stft = self.stft.to(device)
    #     self.device = device
    #     return self

    # def apply(self, fn: callable):
    #     # print('apply')
    #     fn(self.ranges_parameters)
    #     fn(self.stft)
    #     return self

    # def cpu(self):
    #     # print('cpu')
    #     self.ranges_parameters = self.ranges_parameters.cpu()
    #     self.stft = self.stft.cpu()
    #     self.device = "cpu"
    #     return self

    # def cuda(self, device: Optional[Union[int, torch.device]] = None):
    #     # print('cuda')
    #     self.ranges_parameters = self.ranges_parameters.cuda(device)
    #     self.stft = self.stft.cuda(device)
    #     return self


class Filter(DDAFX):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        if ranges_parameters is None:
            ranges_parameters = [[20, 20000], [0.1, 3]]

        self.name = "Filter"
        self.params_names = ["Frequency", "Q"]

        super().__init__(
            num_parameters=2,
            ranges_parameters=ranges_parameters,
            samplerate=samplerate,
        )

    def denormalize_parameters(self, w: Tensor):

        v = torch.zeros_like(w)

        m = torch.log(self.ranges_parameters[:, 0].reshape(1, 2))
        M = torch.log(self.ranges_parameters[:, 1].reshape(1, 2))
        v = torch.exp(w * (M - m) + m)

        return v

    def process(self, x: Tensor, p: Tensor):

        b, a = self.get_coeffs(p)

        x = x.squeeze(1)

        # y = torchaudio.functional.lfilter(
        #     x, a_coeffs=a, b_coeffs=b, clamp=False, batching=True
        # )
        y = fftfilt(x, a_coeffs=a, b_coeffs=b)
        y = y.unsqueeze(1)

        return y

    def get_coeffs(self, v: Tensor) -> Tensor:
        raise NotImplementedError("please implement a get_coeffs method")


class FilterChain(DDAFXChain):
    def __init__(self, *effects_args: Filter, samplerate: float = 44100):
        for fx in effects_args:
            if not isinstance(fx, (Filter, Band)):
                raise TypeError("Effect should be a filter")

        super().__init__(*effects_args, samplerate=samplerate)
        self.effects_list: list[Union[Filter, Band]]

    def __call__(self, x: Tensor, z: Tensor):
        a_list: list[Tensor] = []
        b_list: list[Tensor] = []

        control_idx = 0

        for fx in self.effects_list:
            z_part = z[:, control_idx : control_idx + fx.num_parameters]
            v_part = fx.denormalize_parameters(z_part)
            b, a = fx.get_coeffs(v_part)

            a_list.append(a)
            b_list.append(b)

            control_idx += fx.num_parameters

        x = x.clone().squeeze(1)
        y = chain_fftfilt(x, a_coeffs_list=a_list, b_coeffs_list=b_list)
        y = y.unsqueeze(1)

        return y


class Peak(Band):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v: Tensor):
        batch_size = v.size(0)
        f0 = v[:, 0]
        g_dB = v[:, 1]
        Q = v[:, 2]

        A = torch.pow(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = torch.sin(w0) / (2 * Q)

        a = torch.zeros(batch_size, 3).to(v.device)
        b = torch.zeros(batch_size, 3).to(v.device)

        b[:, 0] = 1 + alpha * A
        b[:, 1] = -2 * torch.cos(w0)
        b[:, 2] = 1 - alpha * A
        a[:, 0] = 1 + alpha / A
        a[:, 1] = -2 * torch.cos(w0)
        a[:, 2] = 1 - alpha / A

        return b, a


class LowShelf(Band):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v: Tensor):
        batch_size = v.size(0)
        f0 = v[:, 0]
        g_dB = v[:, 1]
        Q = v[:, 2]

        A = torch.pow(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = torch.sin(w0) / (2 * Q)

        a = torch.zeros(batch_size, 3).to(v.device)
        b = torch.zeros(batch_size, 3).to(v.device)

        b[:, 0] = A * ((A + 1) - (A - 1) * torch.cos(w0) + 2 * torch.sqrt(A) * alpha)
        b[:, 1] = 2 * A * ((A - 1) - (A + 1) * torch.cos(w0))
        b[:, 2] = A * ((A + 1) - (A - 1) * torch.cos(w0) - 2 * torch.sqrt(A) * alpha)
        a[:, 0] = (A + 1) + (A - 1) * torch.cos(w0) + 2 * torch.sqrt(A) * alpha
        a[:, 1] = -2 * ((A - 1) + (A + 1) * torch.cos(w0))
        a[:, 2] = (A + 1) + (A - 1) * torch.cos(w0) - 2 * torch.sqrt(A) * alpha

        return b, a


class HighShelf(Band):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v: Tensor):
        batch_size = v.size(0)
        f0 = v[:, 0]
        g_dB = v[:, 1]
        Q = v[:, 2]

        A = torch.pow(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = torch.sin(w0) / (2 * Q)

        a = torch.zeros(batch_size, 3).to(v.device)
        b = torch.zeros(batch_size, 3).to(v.device)

        b[:, 0] = A * ((A + 1) + (A - 1) * torch.cos(w0) + 2 * torch.sqrt(A) * alpha)
        b[:, 1] = -2 * A * ((A - 1) + (A + 1) * torch.cos(w0))
        b[:, 2] = A * ((A + 1) + (A - 1) * torch.cos(w0) - 2 * torch.sqrt(A) * alpha)
        a[:, 0] = (A + 1) - (A - 1) * torch.cos(w0) + 2 * torch.sqrt(A) * alpha
        a[:, 1] = 2 * ((A - 1) - (A + 1) * torch.cos(w0))
        a[:, 2] = (A + 1) - (A - 1) * torch.cos(w0) - 2 * torch.sqrt(A) * alpha

        return b, a


class HighPass(Filter):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v: Tensor):
        batch_size = v.size(0)
        f0 = v[:, 0]
        Q = v[:, 1]

        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = torch.sin(w0) / (2 * Q)

        a = torch.zeros(batch_size, 3).to(v.device)
        b = torch.zeros(batch_size, 3).to(v.device)

        b[:, 0] = (1 + torch.cos(w0)) / 2
        b[:, 1] = -(1 + torch.cos(w0))
        b[:, 2] = (1 + torch.cos(w0)) / 2
        a[:, 0] = 1 + alpha
        a[:, 1] = -2 * torch.cos(w0)
        a[:, 2] = 1 - alpha

        return b, a


class LowPass(Filter):
    def __init__(self, ranges_parameters: Tensor = None, samplerate: float = 44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v: Tensor):
        batch_size = v.size(0)
        f0 = v[:, 0]
        Q = v[:, 1]

        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = torch.sin(w0) / (2 * Q)

        a = torch.zeros(batch_size, 3).to(v.device)
        b = torch.zeros(batch_size, 3).to(v.device)

        b[:, 0] = (1 - torch.cos(w0)) / 2
        b[:, 1] = 1 - torch.cos(w0)
        b[:, 2] = (1 - torch.cos(w0)) / 2
        a[:, 0] = 1 + alpha
        a[:, 1] = -2 * torch.cos(w0)
        a[:, 2] = 1 - alpha

        return b, a
