from .main import DDAFX
import torch
import numpy as np
from torch import Tensor


class SineGenerator(DDAFX):
    def __init__(self, freq: float, samplerate: int = 44100):
        super().__init__(
            num_parameters=1,
            ranges_parameters=[[-1, 1]],
            samplerate=samplerate,
        )
        self.freq = freq

    def process(self, x: Tensor, v: Tensor):
        batchSize, _, numSamples = x.size()
        device = x.device
        t = torch.arange(numSamples, device=device) / self.samplerate
        t.reshape(1, 1, numSamples)

        amp = torch.reshape(v, (batchSize, 1, 1))
        sine = torch.sin(t * 2 * np.pi * self.freq)

        out = sine * amp + x

        return out

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        v = self.denormalize_parameters(w)
        return self.process(x, v)