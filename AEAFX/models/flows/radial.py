import torch
from .utils import DifFunction, Tanh, Identity, TanhLayer, SigmoidLayer, Inverse
from ...utils import safe_log


class AbsModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


def radial_fn(
    z: torch.Tensor,
    z0: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    h: DifFunction,
):
    r = get_r(z, z0)
    return z + beta * h(r + alpha) * (z - z0)


def det(
    z: torch.Tensor,
    z0: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    h: DifFunction,
):
    bs, D = z.size()
    device = z.device

    r = get_r(z, z0)

    out = torch.pow(1 + beta * h(alpha + r), D - 1)
    out = out * (1 + beta * h(alpha + r) + beta * h.deriv(alpha + r) * r)
    return out


def get_r(z, z0):
    r = torch.sqrt(torch.sum(torch.square(z - z0), dim=1, keepdim=True))
    return r


def m(x):
    return torch.log(1 + torch.exp(x))


def adapt(beta, alpha):
    out = -alpha + m(beta)
    return out


class Static(torch.nn.Module):
    def __init__(self, dim, h: DifFunction = Inverse):
        super(Static, self).__init__()
        self.dim = dim

        self.z0 = torch.nn.Parameter(torch.randn((1, self.dim)) / self.dim)
        self.alpha = torch.nn.Parameter(torch.rand((1, 1)) / self.dim)
        self.beta = torch.nn.Parameter(torch.randn(1, 1))
        self.h: DifFunction = h()

    def forward(self, z):
        z0: torch.Tensor = self.z0
        alpha: torch.Tensor = torch.abs(self.alpha)
        beta: torch.Tensor = self.beta
        h: DifFunction = self.h

        alpha = torch.abs(alpha)
        beta = adapt(beta, alpha)

        out = radial_fn(z=z, z0=z0, beta=beta, alpha=alpha, h=h)
        return out

    def det(self, z):
        z0: torch.Tensor = self.z0
        alpha: torch.Tensor = torch.abs(self.alpha)
        beta: torch.Tensor = self.beta
        h: DifFunction = self.h

        alpha = torch.abs(alpha)
        beta = adapt(beta, alpha)

        out = det(z=z, z0=z0, beta=beta, alpha=alpha, h=h)
        return out

    def log_det(self, z):
        return torch.log(self.det(z))

    def forward_and_logdet(self, z: torch.Tensor):
        ld = self.log_det(z)
        z = self(z)
        return z, ld


class Dynamic(torch.nn.Module):
    def __init__(self, dim: int, context_size:int,h: DifFunction = Tanh):
        super().__init__()

        self.get_z0 = torch.nn.Linear(context_size, dim)
        self.get_alpha = torch.nn.Sequential(torch.nn.Linear(context_size, 1), AbsModule())
        self.get_beta = torch.nn.Linear(context_size, 1)
        self.h = h()

    def get_params(self, x: torch.Tensor):
        return self.get_z0(x), self.get_alpha(x), self.get_beta(x)

    def forward(self, x: torch.Tensor, v=torch.Tensor):
        z0, alpha, beta = self.get_params(v)
        beta = adapt(beta=beta, alpha=alpha)

        y = radial_fn(z=x, z0=z0, beta=beta, alpha=alpha, h=self.h)
        return y

    def forward_and_logdet(self, x: torch.Tensor, v=torch.Tensor):
        z0, alpha, beta = self.get_params(v)
        beta = adapt(beta=beta, alpha=alpha)

        ld = safe_log(det(z=x, z0=z0, beta=beta, alpha=alpha, h=self.h))
        y = radial_fn(z=x, z0=z0, beta=beta, alpha=alpha, h=self.h)

        return y, ld
