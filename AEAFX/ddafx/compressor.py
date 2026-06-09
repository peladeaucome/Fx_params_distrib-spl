import torch, torchaudio
from torch import Tensor
from .main import DDAFX
from ..utils import dB20
from .filters.utils import fftfilt


class SimpleCompressor(DDAFX):
    def __init__(self, ranges_parameters=None, samplerate: int = 44100):
        if ranges_parameters is None:
            ranges_parameters = [
                [-30, 0],
                [0.1, 30],
                [1, 10],
                [0, 12],
            ]

        self.name = "Simple Compressor"
        self.params_names = ["Threshold", "Time Constant", "Ratio", "Knee"]

        super().__init__(
            num_parameters=4, ranges_parameters=ranges_parameters, samplerate=samplerate
        )

    def denormalize_parameters(self, w: Tensor):
        ranges = self.ranges_parameters.clone().to(w.device)
        ranges[1] = torch.log(ranges[1])

        m = ranges[:, 0].reshape(1, -1)
        M = ranges[:, 1].reshape(1, -1)
        v = w * (M - m) + m
        v[:, 0] = torch.exp(v[:, 0])

        return super().denormalize_parameters(w)

    def process(self, x: Tensor, v: Tensor):
        batch_size, num_channels, num_samples = x.size()

        threshold_dB = v[:, 0].view(batch_size, 1, 1)
        time_constant = v[:, 1].view(batch_size, 1, 1)
        ratio = v[:, 2].view(batch_size, 1, 1)
        knee_dB = v[:, 3].view(batch_size, 1, 1)

        x_G = dB20(x)

        y_G = torch.where(
            2 * (x_G - threshold_dB) < -knee_dB * torch.ones_like(x),
            input=x_G,
            other=torch.where(
                torch.abs(2 * (x_G - threshold_dB)) <= knee_dB * torch.ones_like(x),
                input=x_G
                + (1 / ratio - 1)
                * torch.square(x_G - threshold_dB + knee_dB / 2)
                / (2 * knee_dB),
                other=threshold_dB + (x_G - threshold_dB) / ratio,
            ),
        )

        x_L = x_G - y_G

        del x_G
        del y_G

        alpha = torch.exp(-1 / (time_constant * 0.001 * self.samplerate))
        b = torch.zeros((batch_size, 2), device=self.device)
        b[:, 0] = (1 - alpha)[:, 0, 0]
        a = torch.ones((batch_size, 2), device=self.device)
        a[:, 1] = -alpha[:, 0, 0]

        x_L = x_L.squeeze(1)
        # y_L = torchaudio.functional.lfilter(
        #     x_L, a_coeffs=a, b_coeffs=b, batching=True, clamp=False
        # )
        y_L = fftfilt(x_L, a_coeffs=a, b_coeffs=b)

        y_L = y_L.unsqueeze(1)

        c = torch.pow(10, -y_L / 20)

        return x * c
