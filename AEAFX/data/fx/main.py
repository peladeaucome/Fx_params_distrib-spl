import numpy as np


class DAFx:
    def __init__(self, num_parameters, ranges_parameters, samplerate: float = 44100):
        self.ranges_parameters: np.ndarray = np.array(ranges_parameters)
        self.num_parameters = num_parameters
        self.samplerate = samplerate

    def denormalize_parameters(self, w: np.ndarray):
        m = self.ranges_parameters[:, 0]
        M = self.ranges_parameters[:, 1]

        v = w * (M - m) + m
        return v

    def normalize_parameters(self, v: np.ndarray):
        m = self.ranges_parameters[:, 0]
        M = self.ranges_parameters[:, 1]

        v = (v - m) / (M - m)
        return v

    def process(x, v: np.ndarray):
        return x

    def __call__(self, x: np.ndarray, w: np.ndarray):
        v = self.denormalize_parameters(w)
        x = x / np.amax(np.abs(x))
        y = self.process(x, v)
        return y


class ParameterLess_Dafx(DAFx):
    def __init__(self, samplerate):
        super().__init__(num_parameters=0, ranges_parameters=[], samplerate=samplerate)

    def process(x):
        return x

    def __call__(self, x: np.ndarray):
        y = self.process(x)
        return y


class DAFx_Series(DAFx):
    def __init__(self, *fx_list: DAFx, samplerate=44100):
        self.ranges_parameters = None
        self.num_parameters = 0
        self.fx_list: list[DAFx] = []
        self.samplerate = samplerate

        for fx in fx_list:
            self.append(fx)

    def append(self, fx: DAFx):
        self.fx_list.append(fx)

        if fx.num_parameters > 0:
            if self.ranges_parameters is None:
                self.ranges_parameters = fx.ranges_parameters
            else:
                new_ranges_parameters = np.zeros(
                    (self.num_parameters + fx.num_parameters, 2)
                )
                new_ranges_parameters[: self.num_parameters, :] = self.ranges_parameters
                new_ranges_parameters[self.num_parameters :, :] = fx.ranges_parameters

                self.ranges_parameters = new_ranges_parameters
            self.num_parameters += fx.num_parameters

    def append_multiple(self, *fx_list):
        for fx in fx_list:
            self.append(fx)
    
    def __call__(self, x: np.ndarray, w: np.ndarray):
        y = self.process(x, w)
        return y

    def process(self, x: np.ndarray, w: np.ndarray):
        param_idx = 0
        for fx in self.fx_list:
            if fx.num_parameters > 0:
                x = fx(x, w[param_idx : param_idx + fx.num_parameters])
                param_idx += fx.num_parameters
            else:
                x = fx(x)
        return x


class Peak_Norm(ParameterLess_Dafx):
    def __init__(self, level_dB, samplerate):
        super().__init__(samplerate=samplerate)

        self.level = np.power(10, level_dB / 20)

    def process(self, x):
        x = x - np.mean(x)
        x = x / np.amax(np.abs(x))
        return x * self.level


class RMS_Norm(ParameterLess_Dafx):
    def __init__(self, level_dB, samplerate):
        super().__init__(samplerate=samplerate)

        self.level = np.power(10, level_dB / 20)

    def process(self, x):
        x = x - np.mean(x)
        x = x / np.std(x)
        return x * self.level
