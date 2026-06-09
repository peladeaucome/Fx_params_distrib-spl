from .main import DAFx, ParameterLess_Dafx
import numpy as np
import scipy.signal

# from numba import jit


class Band(DAFx):
    def __init__(self, ranges_parameters=None, samplerate=44100):
        if ranges_parameters is None:
            ranges_parameters = [[20, 20000], [-10, 10], [0.1, 3]]
        super().__init__(
            num_parameters=3,
            ranges_parameters=ranges_parameters,
            samplerate=samplerate,
        )

    def denormalize_parameters(self, w: np.ndarray) -> np.ndarray:
        v = np.zeros_like(w)
        m = self.ranges_parameters[:, 0]
        M = self.ranges_parameters[:, 1]

        v[0] = np.exp(w[0] * (np.log(M[0]) - np.log(m[0])) + np.log(m[0]))
        v[1] = w[1] * (M[1] - m[1]) + m[1]
        v[2] = w[2] * (M[2] - m[2]) + m[2]
        return v

    def get_coeffs(self, v):
        a = np.ones(1)
        b = np.ones(1)
        return b, a

    # @jit
    def process(self, x: np.ndarray, v: np.ndarray):
        b, a = self.get_coeffs(v)
        y = scipy.signal.lfilter(b=b, a=a, x=x)
        return y


class Filter(DAFx):
    def __init__(self, ranges_parameters=None, samplerate=44100):
        if ranges_parameters is None:
            ranges_parameters = [[20, 20000], [0.1, 3]]
        super().__init__(
            num_parameters=2,
            ranges_parameters=ranges_parameters,
            samplerate=samplerate,
        )

    def denormalize_parameters(self, w: np.ndarray) -> np.ndarray:
        v = np.zeros_like(w)
        m = self.ranges_parameters[:, 0]
        M = self.ranges_parameters[:, 1]

        v[0] = np.exp(w[0] * (np.log(M[0]) - np.log(m[0])) + np.log(m[0]))
        v[1] = w[1] * (M[1] - m[1]) + m[1]
        return v

    def get_coeffs(self, v):
        a = np.ones(1)
        b = np.ones(1)
        return b, a

    # @jit
    def process(self, x: np.ndarray, v: np.ndarray):
        b, a = self.get_coeffs(v)
        y = scipy.signal.lfilter(b=b, a=a, x=x)
        return y


class HighPass(Filter):
    def get_coeffs(self, v):
        f0 = v[0]
        Q = v[1]

        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = (1 + np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        b = np.array([b0, b1, b2])
        a = np.array([a0, a1, a2])
        return b, a


class Peak(Band):
    def __init__(self, ranges_parameters: np.ndarray = None, samplerate=44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v):
        f0 = v[0]
        g_dB = v[1]
        Q = v[2]

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        return b, a


class LowShelf(Band):
    def __init__(self, ranges_parameters=None, samplerate=44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v):
        f0 = v[0]
        g_dB = v[1]
        Q = v[2]

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a[2] = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return b, a


class HighShelf(Band):
    def __init__(self, ranges_parameters=None, samplerate=44100):
        super().__init__(ranges_parameters, samplerate)

    def get_coeffs(self, v):
        f0 = v[0]
        g_dB = v[1]
        Q = v[2]

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a[2] = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return b, a


class FixedBand(DAFx):
    def __init__(
        self, f0: float, gain_range: list[float], Q: float, samplerate: int = 44100
    ):
        super().__init__(
            num_parameters=1,
            ranges_parameters=[gain_range],
            samplerate=samplerate,
        )
        self.f0 = f0
        self.Q = Q

    def get_coeffs(self, v):
        a = np.ones(1)
        b = np.ones(1)
        return b, a

    def process(self, x: np.ndarray, v: np.ndarray):
        b, a = self.get_coeffs(v)
        y = scipy.signal.lfilter(b=b, a=a, x=x)
        return y


class FixedPeak(FixedBand):
    def __init__(self, f0, gain_range, Q, samplerate=44100):
        super().__init__(f0, gain_range, Q, samplerate)

    def get_coeffs(self, v):
        f0 = self.f0
        g_dB = v[0]
        Q = self.Q

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        return b, a


class FixedLowShelf(FixedBand):
    def __init__(self, f0, gain_range, Q, samplerate=44100):
        super().__init__(f0, gain_range, Q, samplerate)

    def get_coeffs(self, v):
        f0 = self.f0
        g_dB = v[0]
        Q = self.Q

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a[2] = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return b, a


class FixedHighShelf(FixedBand):
    def __init__(self, f0, gain_range, Q, samplerate=44100):
        super().__init__(f0, gain_range, Q, samplerate)

    def get_coeffs(self, v):
        f0 = self.f0
        g_dB = v[0]
        Q = self.Q

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / self.samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a[2] = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return b, a


class ConstantFilter(ParameterLess_Dafx):
    def __init__(self, b, a, samplerate=44100):
        super().__init__(samplerate=samplerate)
        self.a = a
        self.b = b

    # @jit
    def process(self, x: np.ndarray):
        y = scipy.signal.lfilter(b=self.b, a=self.a, x=x)
        return y


class ConstantPeak(ConstantFilter):
    def __init__(self, f0, g_dB, Q, samplerate=44100):

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        super().__init__(b, a, samplerate)


class ConstantLowShelf(ConstantFilter):
    def __init__(self, f0, g_dB, Q, samplerate=44100):

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a[2] = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        super().__init__(b, a, samplerate)


class ConstantHighShelf(ConstantFilter):
    def __init__(self, f0, g_dB, Q, samplerate=44100):

        A = np.power(10, g_dB / 40)
        w0 = 2 * np.pi * f0 / samplerate
        alpha = np.sin(w0) / (2 * Q)

        b = np.zeros(3)
        a = np.zeros(3)

        b[0] = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b[1] = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b[2] = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a[0] = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a[1] = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a[2] = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        super().__init__(b, a, samplerate)
