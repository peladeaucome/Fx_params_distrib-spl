import torch.nn as nn
import torch
from typing import Literal
from torch import Tensor
import math

class Swish(nn.Module):
    def __init__(self, dim: int = None):
        super().__init__()

        if dim is None:
            dim = 1

        self.weigths = nn.Parameter(torch.ones(dim, dtype=torch.float))

    def forward(self, x):
        a = nn.functional.softplus(self.weigths)
        a = self.weigths
        return x * nn.functional.sigmoid(x * a)


class FCBlock(nn.Sequential):
    def __init__(
        self,
        input_dim,
        output_dim,
        nl_type: Literal["none", "swish", "silu"] = "swish",
        bn: bool = True,
        dropout: float = 0,
    ):
        super().__init__()

        # if bn:
        #     self.append(nn.LayerNorm([input_dim]))
        if dropout is not None:
            self.append(nn.Dropout(p=dropout))
        self.append(nn.Linear(input_dim, output_dim))
        if bn:
            self.append(nn.BatchNorm1d(output_dim))
            # self.append(LayerNormalNorm(output_dim))

        if nl_type == "silu":
            self.append(nn.SiLU())
        elif nl_type == "swish":
            self.append(Swish(output_dim))


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        nl_type: Literal["swish", "silu"] = "swish",
        bn: bool = True,
        dropout: float = 0.0,
    ):
        if hidden_dim is None:
            hidden_dim = dim
        super().__init__()
        self.fc1 = FCBlock(
            input_dim=dim,
            output_dim=hidden_dim,
            nl_type=nl_type,
            bn=bn,
            dropout=dropout,
        )
        self.lin2 = nn.Sequential(
            # nn.LayerNorm([hidden_dim]),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
        )

        if bn:
            self.lin2.append(nn.BatchNorm1d(dim))
            # self.lin2.append(LayerNormalNorm(dim))

        if nl_type == "silu":
            self.nl2 = nn.SiLU()
        elif nl_type == "swish":
            self.nl2 = Swish(dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = self.lin2(y)
        out = self.nl2(y + x)
        return out


## From https://github.com/DanielEftekhari/normality-normalization/tree/main
class LayerNormalNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 noise_train=1.0,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(LayerNormalNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, (list, tuple)):
            normalized_shape = torch.Size(normalized_shape)
        self.normalized_shape = normalized_shape
        self._dims = [-(i+1) for i in range(len(self.normalized_shape))]
        self.elementwise_affine = elementwise_affine
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        # cached variables stored here
        self._init_cache()

        if self.elementwise_affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.normalized_shape).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.normalized_shape).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        if self.normalized_shape != x.shape[-len(self.normalized_shape):]:
            raise ValueError('input shape {} inconsistent with normalized_shape {}'.format(x.shape, self.normalized_shape))

    def _init_cache(self):
        self.cache = {'lmbda_estimate': None}

    def _reset_cache(self):
        self.cache.clear()
        self._init_cache()

    def forward(self, x):
        self._reset_cache()
        self._check_input_dim(x)

        mean = torch.mean(x, dim=self._dims, keepdim=True)
        var = torch.var(x, correction=0, dim=self._dims, keepdim=True)
        x = self._standardize(x, mean, var)

        x_sign = torch.sign(x)
        x_sign[x_sign == 0] = 1.
        x_abs = torch.abs(x)

        lmbda = self._estimate(x, x_sign, x_abs)
        self.cache['lmbda_estimate'] = lmbda.detach().clone()
        x = self._transform(x_sign, x_abs, lmbda)

        with torch.no_grad():
            mean_ = torch.mean(x, dim=self._dims, keepdim=True)
            norm_cast = torch.mean(torch.abs(x - mean_), dim=self._dims, keepdim=True)
        if self.training:
            x = scaled_additive_normal_noise(x, norm_cast, mean=0., std=self.noise_train)

        if self.elementwise_affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _estimate(self, x, x_sign, x_abs, order=2):
        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones(size=self.normalized_shape).to(x_abs.device) - d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones(size=self.normalized_shape).to(x_abs.device) - d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=self._dims, keepdim=True)
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=self._dims, keepdim=True)
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=self._dims, keepdim=True)
            d_sub_avg = d - dmean
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=self._dims, keepdim=True)
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain + shift
    
def scaled_additive_normal_noise(x, scale, mean=0., std=1.):
    x = x + (torch.randn_like(x) * scale * std + mean)
    return x
