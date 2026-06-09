from .main import DAFx, ParameterLess_Dafx
from .utils import dB20
import numpy as np
from numba import njit


class Compressor(DAFx):
    def __init__(self, ranges_parameters=None, samplerate: float = 44100):
        if ranges_parameters is None:
            ranges_parameters = [
                [-20, 0],  # Threshold
                [1, 10],  # Ratio
                [0, 12],  # Knee
                [0.1, 100],  # Attack
                [1, 500],  # Release
            ]
        super().__init__(
            num_parameters=5, ranges_parameters=ranges_parameters, samplerate=samplerate
        )

    def process(self, x, v):

        threshold_dB = v[0]
        ratio = v[1]
        knee_dB = v[2]
        attack_ms = v[3]
        release_ms = v[4]

        y = compress(
            x,
            threshold_dB=threshold_dB,
            ratio=ratio,
            knee_dB=knee_dB,
            attack_ms=attack_ms,
            release_ms=release_ms,
            samplerate=self.samplerate,
        )
        return y


@njit
def compress(
    x: np.ndarray,
    threshold_dB: float,
    ratio: float,
    knee_dB: float,
    attack_ms: float,
    release_ms: float,
    samplerate: int = 44100,
):
    x_G = 20 * np.log10(np.abs(x) + 1e-6)

    num_samples = len(x)

    alpha_attack = np.exp(-1 / (attack_ms * 0.001 * samplerate))
    alpha_release = np.exp(-1 / (release_ms * 0.001 * samplerate))

    y_G = np.zeros(np.shape(x_G))

    where_result = np.where(2 * (x_G - threshold_dB) < -knee_dB)
    y_G[where_result] = x_G[where_result]

    # y_G = x_G #Skipping the first_condition to gain time

    if knee_dB > 0:
        where_result = np.where(2 * np.abs(x_G - threshold_dB) <= knee_dB)
        y_G[where_result] = x_G[where_result] + (1 / ratio - 1) * np.square(
            x_G[where_result] - threshold_dB + knee_dB / 2
        ) / (
            2 * knee_dB
        )  # Middle condition

    where_result = np.where(2 * (x_G - threshold_dB) > knee_dB)
    y_G[where_result] = threshold_dB + (x_G[where_result] - threshold_dB) / ratio

    x_L = x_G - y_G

    y_L = np.zeros(num_samples)

    y_L[0] = 0

    for n in range(1, num_samples):
        if x_L[n] > y_L[n - 1]:
            y_L[n] = alpha_attack * y_L[n - 1] + (1 - alpha_attack) * x_L[n]
        else:
            y_L[n] = alpha_release * y_L[n - 1] + (1 - alpha_release) * x_L[n]

    c = np.power(10, -y_L * 0.05)
    return x * c


@njit
def compute_envelope_single(
    x: np.ndarray, t_att: float, t_rel: float, p=2, samplerate=16000
):
    y = np.zeros(x.shape)

    alpha_att = 1 - np.exp(-1 / (t_att * samplerate))
    alpha_rel = 1 - np.exp(-1 / (t_rel * samplerate))

    prev = 0
    for n in range(len(x)):
        x_n_pow = np.power(np.abs(x[n]), p)
        if x_n_pow > prev:
            alpha = alpha_att
        else:
            alpha = alpha_rel

        prev = alpha * x_n_pow + (1 - alpha) * prev
        y[n] = prev
    return y


@njit
def compute_loudness_single(
    x: np.ndarray, tshort_att=0.001, tshort_rel=0.01, s=10, samplerate=16000, p=2
):
    tlong = max(tshort_att, tshort_rel) * s

    a_short = compute_envelope_single(
        x, t_att=tshort_att, t_rel=tshort_rel, p=p, samplerate=samplerate
    )
    a_short = np.abs(a_short + 1e-6)
    a_long = compute_envelope_single(
        x, t_att=tlong, t_rel=tlong, p=p, samplerate=samplerate
    )
    a_long = np.abs(a_long + 1e-6)
    L = a_short / a_long
    return L


class DynamicAdjustment(ParameterLess_Dafx):
    def __init__(
        self,
        tshort_att,
        tshort_rel,
        time_ratio=10,
        comp_ratio=1,
        order=2,
        samplerate=44100,
    ):
        super().__init__(samplerate)
        self.comp_ratio = comp_ratio
        self.time_ratio = time_ratio
        self.order = order
        self.tshort_att = tshort_att
        self.tshort_rel = tshort_rel

    def process(self, x):
        L = compute_loudness_single(
            x,
            self.tshort_att,
            self.tshort_rel,
            self.time_ratio,
            self.samplerate,
            self.order,
        )
        g = np.power(L, (self.comp_ratio - 1) / self.order)
        return g * x
