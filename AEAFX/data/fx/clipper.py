from .main import DAFx
from .utils import dB20
import numpy as np
from numba import njit


class SoftClipper(DAFx):
    def __init__(self, ranges_parameters=None, samplerate: float = 44100):
        if ranges_parameters is None:
            ranges_parameters = [
                [-6, 0],  # Threshold
                [0, 6],  # Knee
            ]
        super().__init__(
            num_parameters=2, ranges_parameters=ranges_parameters, samplerate=samplerate
        )

    def process(self, x:np.ndarray, v:np.ndarray):

        T = v[0]
        W = v[1]

        x_dB = 20 * np.log10(np.abs(x) + 1e-10)

        y_dB = np.zeros(np.shape(x_dB))

        # wh_idx = np.where(2 * (x_dB - T) < -W)
        y_dB = x_dB

        # y_G = x_G #Skipping the first_condition to gain time

        if W > 0:
            # Middle condition
            wh_idx = np.where(2 * np.abs(x_dB - T) <= W)
            x_dB_w = x_dB[wh_idx]
            y_dB[wh_idx] = x_dB_w - np.square(x_dB_w - T + W / 2) / (2 * W)

        # wh_idx = np.where(2 * (x_dB - T) > W)
        wh_idx = np.where((x_dB - T) > W/2)
        y_dB[wh_idx] = T

        gain = np.power(10, (y_dB - x_dB) * 0.05)

        y = x * gain

        return y

