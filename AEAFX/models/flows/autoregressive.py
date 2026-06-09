import torch
from .utils import Flow
from torch import nn
from torch import Tensor
from ...utils import safe_log, safe_inv
from ..layers import ResBlock, Swish
from typing import Callable
import numpy as np

eps = 1e-6
# eps = 1e-16
# eps = 1e-8
# eps = 0

inv = lambda x: 1 / x


def inv_sigmoid(x: Tensor) -> Tensor:
    # return torch.special.logit(x, eps=1e-6)
    # return safe_log(x, eps=eps) - safe_log(1 - x, eps=eps)
    # return torch.log(x * inv(1 - x))
    return torch.log(x) - torch.log(1 - x)


def sigmoid_deriv(x: Tensor) -> Tensor:
    sig = torch.sigmoid
    return sig(x) * (1 - sig(x))


def inv_sigmoid_deriv(x: Tensor) -> Tensor:
    # return safe_inv(x, eps=eps) + safe_inv(1 - x, eps=eps)
    return inv(x) + inv(1 - x)


class LoTriLinear(nn.Linear):
    def __init__(self, in_features: int, offset: int = 0, bias=True):
        super().__init__(in_features, in_features, bias=bias)
        with torch.no_grad():
            self.weight.data.copy_(torch.zeros(in_features))
            self.weight.copy_(torch.tril(self.weight, offset))
        self.weight.register_hook(
            lambda grad: grad * torch.tril(torch.ones_like(grad), offset)
        )


class TriangleFCBlock(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            LoTriLinear(dim, offset=-1, bias=True),
            Swish(dim),
            LoTriLinear(dim, offset=-1, bias=True),
            Swish(dim),
        )


class Adaptor(nn.Module):
    def __init__(self, dim: int, num_chans: int):
        super().__init__()

        self.dim = dim
        self.num_chans = num_chans
        self.weight = nn.Parameter(torch.randn(1, dim, num_chans))
        # self.bias = nn.Parameter(torch.randn(1, dim, num_chans))

    def forward(self, x: Tensor):
        return x.unsqueeze(2) * self.weight


class DSF_Static(Flow):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.get_a_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )
        self.get_b_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )
        self.get_w_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )

        # self.a_adapt = nn.Parameter(torch.randn(1, dim, hidden_dim))
        # self.b_adapt = nn.Parameter(torch.randn(1, dim, hidden_dim))
        # self.w_adapt = nn.Parameter(torch.randn(1, dim, hidden_dim))

        self.a_bias = nn.Parameter(torch.randn(1, dim, hidden_dim) / hidden_dim)
        self.b_bias = nn.Parameter(torch.randn(1, dim, hidden_dim) / hidden_dim)
        self.w_bias = nn.Parameter(torch.randn(1, dim, hidden_dim) / hidden_dim)
        self.scale = 1 - 1e-4

    def scale_fn(self, x: Tensor):
        return ((x - 0.5) * self.scale) + 0.5

    def get_a_b_w(self, z: Tensor, c: Tensor):
        bs, dim = z.size()
        # z = z.unsqueeze(2).contiguous()

        a_ar = self.get_a_from_z(z)
        b_ar = self.get_b_from_z(z)
        w_ar = self.get_w_from_z(z)

        ###############################################
        a = a_ar + self.a_bias
        b = b_ar + self.b_bias
        w = w_ar + self.w_bias

        ###############################################

        a = nn.functional.softplus(a)
        w = torch.softmax(w, dim=2)

        return a, b, w

    def forward(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple

        a, b, w = self.get_a_b_w(z, c)

        z = z.unsqueeze(2)

        out = torch.sigmoid(z * a + b)
        out = inv_sigmoid(self.scale_fn((out * w).sum(2)))

        return out

    def logdet(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple

        a, b, w = self.get_a_b_w(z, c)

        z = z.unsqueeze(2)

        zab = z * a + b
        zab_sig = torch.sigmoid(zab)

        det1 = (w * a * sigmoid_deriv(zab)).sum(2) * self.scale
        det2 = inv_sigmoid_deriv(self.scale_fn((zab_sig * w).sum(2)))

        ld = (safe_log(det1, eps=eps) + safe_log(det2, eps=eps)).sum(1, keepdim=True)

        return ld

    def forward_and_logdet(self, *z_tuple: Tensor) -> tuple[Tensor]:
        z, c = z_tuple

        a, b, w = self.get_a_b_w(z, c)

        z = z.unsqueeze(2)

        zab = z * a + b
        zab_sig = torch.sigmoid(zab)

        out = inv_sigmoid(self.scale_fn((zab_sig * w).sum(2)))
        det1 = (w * a * sigmoid_deriv(zab)).sum(2) * self.scale
        det2 = inv_sigmoid_deriv(self.scale_fn((zab_sig * w).sum(2)))

        ld = (safe_log(det1, eps=eps) + safe_log(det2, eps=eps)).sum(1, keepdim=True)
        return out, ld


class DSF_Dynamic(DSF_Static):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        context_size: int,
    ):
        super().__init__(dim=dim, hidden_dim=hidden_dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

        bn = False

        self.get_a_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )
        self.get_b_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )
        self.get_w_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )

    def get_a_b_w(self, z: Tensor, c: Tensor):
        bs, dim = z.size()
        # z = z.unsqueeze(2).contiguous()

        ###############################################
        a = self.get_a_from_z(z) + self.get_a_from_c(c) + self.a_bias
        b = self.get_b_from_z(z) + self.get_b_from_c(c) + self.b_bias
        w = self.get_w_from_z(z) + self.get_w_from_c(c) + self.w_bias

        ###############################################

        a = nn.functional.softplus(a)

        w = torch.softmax(w, dim=2)

        return a, b, w


class IAF_Static(Flow):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

        self.get_a_from_z = nn.Sequential(
            TriangleFCBlock(dim), LoTriLinear(dim, offset=-1, bias=False)
        )
        self.get_b_from_z = nn.Sequential(
            TriangleFCBlock(dim), LoTriLinear(dim, offset=-1, bias=False)
        )

        self.a_bias = nn.Parameter(torch.randn(1, dim))
        self.b_bias = nn.Parameter(torch.randn(1, dim))

    def get_a_b(self, z: Tensor) -> tuple[Tensor]:
        bs, dim = z.size()

        a_ar = self.get_a_from_z(z)
        b_ar = self.get_b_from_z(z)

        ###############################################
        a = a_ar + self.a_bias
        b = b_ar + self.b_bias

        ###############################################
        # a = -torch.nn.functional.softplus(a) + 2
        a = torch.tanh(a) * 2
        # b = b
        return a, b

    def forward(self, *z_tuple: Tensor) -> Tensor:
        z = z_tuple[0]

        a, b = self.get_a_b(z)

        out = a.exp() * z + b

        return out

    def logdet(self, *z_tuple: Tensor) -> Tensor:
        z = z_tuple[0]

        a, b = self.get_a_b(z)

        ld = a.sum(1, keepdim=True)

        return ld

    def forward_and_logdet(self, *z_tuple: Tensor) -> tuple[Tensor]:
        z = z_tuple[0]

        a, b = self.get_a_b(z)

        out = a.exp() * z + b

        ld = a.sum(1, keepdim=True)
        return out, ld


class IAF_Dynamic(Flow):
    def __init__(
        self,
        dim: int,
        context_size: int,
    ):
        super().__init__()
        self.dim = dim

        bn = False
        self.get_a_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            LoTriLinear(dim, offset=-1, bias=False),
        )
        self.get_b_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            LoTriLinear(dim, offset=-1, bias=False),
        )

        self.get_a_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, dim, bias=False),
        )
        self.get_b_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, dim, bias=False),
        )

        self.a_bias = nn.Parameter(torch.randn(1, dim))
        self.b_bias = nn.Parameter(torch.randn(1, dim))

    def get_a_b(self, z: Tensor, c: Tensor) -> tuple[Tensor]:
        bs, dim = z.size()
        ###############################################
        a = self.get_a_from_z(z) + self.get_a_from_c(c) + self.a_bias
        b = self.get_b_from_z(z) + self.get_b_from_c(c) + self.b_bias

        ###############################################
        # b = b
        return a, b

    def forward(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple

        a, b = self.get_a_b(z, c)
        a_sig = torch.sigmoid(a)

        out = a_sig * z + (1 - a_sig) * b

        return out

    def logdet(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple

        a, b = self.get_a_b(z, c)

        ld = -torch.nn.functional.softplus(-a).sum(1, keepdim=True)

        return ld

    def forward_and_logdet(self, *z_tuple: Tensor) -> tuple[Tensor]:
        z, c = z_tuple

        a, b = self.get_a_b(z, c)

        a, b = self.get_a_b(z, c)
        a_sig = torch.sigmoid(a)

        out = a_sig * z + (1 - a_sig) * b
        ld = -torch.nn.functional.softplus(-a).sum(1, keepdim=True)
        return out, ld


class FlowPP_Static(Flow):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.get_pi_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )
        self.get_mu_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )
        self.get_si_from_z = nn.Sequential(
            TriangleFCBlock(dim),
            Adaptor(dim, hidden_dim),
        )

        self.pi_bias = nn.Parameter(torch.zeros(1, dim, hidden_dim))
        self.mu_bias = nn.Parameter(torch.randn(1, dim, hidden_dim))
        self.si_bias = nn.Parameter(torch.zeros(1, dim, hidden_dim))

    def get_params(self, z: Tensor, c) -> tuple[Tensor]:
        bs, dim = z.size()
        # z = z.unsqueeze(2).contiguous()

        pi = self.get_pi_from_z(z) + self.pi_bias
        mu = self.get_mu_from_z(z) + self.mu_bias
        si = self.get_si_from_z(z) + self.si_bias

        ###############################################

        si = torch.nn.functional.softplus(si) + 0.7
        pi = torch.softmax(pi, dim=2)

        return mu, si, pi

    @staticmethod
    def gauss(x: Tensor, mu: Tensor, si: Tensor):
        # print(x.size(), mu.size(), si.size())
        return (-((x - mu) / si).square() * 0.5).exp() / (si * np.sqrt(np.pi * 2))

    def gauss_der(self, x: Tensor, mu: Tensor, si: Tensor):
        return -(x - mu) / (si.square()) * self.gauss(x, mu, si)

    def forward(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple
        mu, si, pi = self.get_params(z, c)

        g = (pi * self.gauss(z.unsqueeze(2), mu, si)).sum(2)

        z = z + g

        return z

    def logdet(self, *z_tuple: Tensor) -> Tensor:
        z, c = z_tuple

        mu, si, pi = self.get_params(z, c)

        dg: Tensor = (pi * self.gauss_der(z.unsqueeze(2), mu, si)).sum(2)
        ld = (dg + 1).log()
        ld = ld.sum(1, keepdim=True)
        return ld

    def forward_and_logdet(self, *z_tuple: Tensor) -> tuple[Tensor]:
        z, c = z_tuple
        mu, si, pi = self.get_params(z, c)

        g = (pi * self.gauss(z.unsqueeze(2), mu, si)).sum(2)

        z = z + g

        dg: Tensor = (pi * self.gauss_der(z.unsqueeze(2), mu, si)).sum(2)
        ld = (dg + 1).log()
        ld = ld.sum(1, keepdim=True)

        return z, ld


class FlowPP_Dynamic(FlowPP_Static):
    def __init__(self, dim: int, hidden_dim: int, context_size):
        super().__init__(dim, hidden_dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.get_pi_from_c = nn.Sequential(
            ResBlock(context_size, bn=False),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )
        self.get_mu_from_c = nn.Sequential(
            ResBlock(context_size, bn=False),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )
        self.get_si_from_c = nn.Sequential(
            ResBlock(context_size, bn=False),
            nn.Linear(context_size, dim * hidden_dim, bias=False),
            nn.Unflatten(1, (dim, hidden_dim)),
        )

    def get_params(self, z: Tensor, c: Tensor) -> tuple[Tensor]:
        pi = self.get_pi_from_z(z) + self.get_pi_from_c(c) + self.pi_bias
        mu = self.get_mu_from_z(z) + self.get_mu_from_c(c) + self.mu_bias
        si = self.get_si_from_z(z) + self.get_si_from_c(c) + self.si_bias

        pi = torch.softmax(pi, dim=2)
        si = torch.nn.functional.softplus(si) + 0.5

        return mu, si, pi


class SoftplusWithEps(nn.Softplus):
    def __init__(self, beta=1, threshold=20, eps=1):
        super().__init__(beta, threshold)
        self.eps = eps

    def forward(self, input):
        return super().forward(input) + self.eps
