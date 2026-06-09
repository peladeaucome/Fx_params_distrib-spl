import numpy as np


def dB20(x:np.ndarray, eps:float=1e-6):
    x_abs = np.maximum(np.abs(x), eps)
    return 20*np.log10(x_abs)


