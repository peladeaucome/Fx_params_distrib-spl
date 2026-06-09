import torch
from .main import DDAFX
from ..utils import dB20, idB20


class SoftClipper(DDAFX):
    """Soft clipper object which is a dynamic range compressor with instantaneous attack and release times."""

    def __init__(
        self, ranges_parameters: torch.Tensor = None, samplerate: float = 44100
    ):
        if ranges_parameters is None:
            ranges_parameters = [[-20, 0], [0, 12]]

        self.name = "Soft Clipper"
        self.params_names = ["Threshold", "Knee"]

        super().__init__(
            num_parameters=2, ranges_parameters=ranges_parameters, samplerate=samplerate
        )

    def process(self, x: torch.Tensor, v: torch.Tensor):
        batch_size = x.size(0)
        threshold_dB = v[:, 0].reshape(batch_size, 1, 1)
        knee_dB = v[:, 1].reshape(batch_size, 1, 1)

        x_dB = dB20(x)

        y_dB = torch.where(
            2 * (x_dB - threshold_dB) < -knee_dB * torch.ones_like(x),
            input=x_dB,
            other=torch.where(
                torch.abs(2 * (x_dB - threshold_dB)) <= knee_dB * torch.ones_like(x),
                input=x_dB
                - torch.square(x_dB - threshold_dB + knee_dB / 2)
                / (2 * knee_dB),
                other=threshold_dB,
            ),
        )

        g_dB = y_dB-x_dB
        g = idB20(g_dB)
        y = x*g
        
        return y


