import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class Swish(nn.Module):
    def __init__(self, dim: int = None):
        super().__init__()

        if dim is None:
            dim = 1

        self.weigths = nn.Parameter(torch.ones(1, dim, 1, dtype=torch.float))

    def forward(self, x):
        a = nn.functional.softplus(self.weigths)
        a = self.weigths
        return x * nn.functional.sigmoid(x * a)


class Swish2d(nn.Module):
    def __init__(self, dim: int = None):
        super().__init__()

        if dim is None:
            dim = 1

        self.weigths = nn.Parameter(torch.ones(1, dim, 1, 1, dtype=torch.float))

    def forward(self, x):
        a = nn.functional.softplus(self.weigths)
        a = self.weigths
        return x * nn.functional.sigmoid(x * a)


class SelfAttention(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.query = nn.Linear(chan, chan)
        self.key = nn.Linear(chan, chan)
        self.val = nn.Linear(chan, chan)

        self.token = torch.nn.Parameter(torch.randn(1, chan, 1))
        self.posencoding_freqs = torch.nn.Parameter(torch.randn(1, chan, 1))

    def forward(self, x: Tensor):
        bs, chan, N = x.size()
        device=x.device
        token = self.token.expand(x.size(0), chan, 1)

        x = torch.cat((x, token), dim=2)
        posenc = torch.sin(self.posencoding_freqs*(torch.arange(N+1, device=device).unsqueeze(0).unsqueeze(1)))
        x=x+posenc

        x = x.swapdims(-1, -2)
        q: Tensor = self.query(x)  # (*,T,E)
        k: Tensor = self.key(x)  # (*,T,E)
        v: Tensor = self.val(x)  # (*,T,E)

        mat_mul = q @ k.transpose(-1, -2)  # (*,T, E) (*,E, T) --> (*,T,T)
        att = mat_mul / np.sqrt(k.size(-1))
        att = torch.nn.functional.softmax(att, dim=-1)
        h = att @ v

        h = h[:, -1, :]
        return h
    
class ResMultiHeadAttention(nn.MultiheadAttention):
    def forward(self, x):
        return x+super().forward(x, x, x)


class Frontend(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim


class SimpleFrontend(Frontend):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(out_dim=out_channels)
        self.conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bn = torch.nn.BatchNorm1d(num_features=out_channels)
        self.nl = Swish(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.nl(self.bn(self.conv(input))), axis=2)
