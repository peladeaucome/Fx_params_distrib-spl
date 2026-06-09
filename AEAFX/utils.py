import torch
from torch import Tensor


def safe_log(x: Tensor, eps: float = 1e-10) -> Tensor:
    return torch.log(torch.abs(x) + eps)



def safe_log2(x: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.log2(torch.abs(x) + eps)


def dB20(x: Tensor, eps: float = 1e-6) -> Tensor:
    return 20 * torch.log10(torch.abs(x) + eps)


def idB20(x_dB: Tensor) -> Tensor:
    return torch.pow(10, x_dB / 20)

def safe_inv(x:Tensor, eps: float = 1e-12) -> Tensor:
    sgn = x.sign()
    return (1/(torch.abs(x)+eps))*sgn
