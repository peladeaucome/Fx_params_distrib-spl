import torch
from torch import Tensor
import numpy as np


def fftfilt(
    x: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor, pad=4096
) -> Tensor:
    bs, N = x.size()
    n_fft = int(np.power(2, np.ceil(np.log2(N + pad))))
    x_fft = torch.fft.rfft(x, n=n_fft, dim=1)

    a_fft = torch.fft.rfft(a_coeffs, n=n_fft, dim=-1)
    b_fft = torch.fft.rfft(b_coeffs, n=n_fft, dim=-1)
    h_fft = b_fft / a_fft

    y_fft = x_fft * h_fft
    y: Tensor = torch.fft.irfft(y_fft, dim=1)
    # y = y[:, :N]
    y, _ = torch.split(y, (N, n_fft - N), dim=-1)
    return y


def chain_fftfilt(
    x: torch.Tensor,
    a_coeffs_list: list[torch.Tensor],
    b_coeffs_list: list[torch.Tensor],
    pad: int = 4096,
) -> Tensor:
    bs, N = x.size()
    n_fft = int(np.power(2, np.ceil(np.log2(N + pad))))
    x_fft = torch.fft.rfft(x, n=n_fft, dim=-1)

    num_filters = len(a_coeffs_list)
    for i in range(num_filters):
        a_coeffs = a_coeffs_list[..., i]
        b_coeffs = b_coeffs_list[..., i]
        a_fft = torch.fft.rfft(a_coeffs, n=n_fft, dim=-1)
        b_fft = torch.fft.rfft(b_coeffs, n=n_fft, dim=-1)
        h_fft = b_fft / a_fft
        x_fft = x_fft * h_fft

    y: Tensor = torch.fft.irfft(x_fft, dim=-1)
    y = y[..., :N]
    return y
