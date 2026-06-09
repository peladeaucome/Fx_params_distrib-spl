import torch
from .utils import DifFunction, Tanh, Identity, TanhLayer, SigmoidLayer
from ...utils import safe_log


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """performs the dot produc of ```a``` and ```b```"""
    return torch.sum(a * b, axis=1, keepdim=True)


def m(x: torch.Tensor) -> torch.Tensor:
    return torch.log(1 + torch.exp(x)) - 1


def planar_fn(
    x: torch.Tensor, w: torch.Tensor, u: torch.Tensor, b: torch.Tensor, h: DifFunction
) -> torch.Tensor:
    uhat = u + (m(dot(w, u)) - dot(w, u)) * w / dot(w, w)

    out = x + uhat * h(dot(w, x) + b)
    return out


def det(
    x: torch.Tensor, u: torch.Tensor, w: torch.Tensor, b: torch.Tensor, h: DifFunction
) -> torch.Tensor:

    uhat = u + (m(dot(w, u)) - dot(w, u)) * w / dot(w, w)

    psi = h.deriv(dot(w, x) + b) * w
    out = torch.abs(1 + dot(uhat, psi))

    return out


def log_det(
    x: torch.Tensor, u: torch.Tensor, w: torch.Tensor, b: torch.Tensor, h: DifFunction
) -> torch.Tensor:
    return safe_log(det(x=x, u=u, w=w, b=b, h=h))


class Static(torch.nn.Module):
    def __init__(self, dim, h=Tanh):
        super(Static, self).__init__()
        self.dim = dim

        self.w = torch.nn.Parameter(torch.randn((1, self.dim)) / self.dim)
        self.u = torch.nn.Parameter(torch.randn((1, self.dim)) / self.dim)
        self.b = torch.nn.Parameter(torch.randn(1, 1))

        self.h = h()

    def forward(self, *z):
        z=z[0]
        u: torch.Tensor = self.u
        w: torch.Tensor = self.w
        b: torch.Tensor = self.b
        h: callable = self.h

        out = planar_fn(x=z, w=w, u=u, b=b, h=h)
        return out

    def det(self, *z):
        z=z[0]
        u: torch.Tensor = self.u
        w: torch.Tensor = self.w
        b: torch.Tensor = self.b

        out = det(x=z, u=u, w=w, b=b, h=self.h)
        return out

    def log_det(self, *z):
        z=z[0]
        return torch.log(self.det(z))

    def flow(self, *z):
        z=z[0]
        det = self.det(z)
        z = self(z)
        return z, det

    def forward_and_logdet(self, *z: torch.Tensor):
        z=z[0]
        ld = self.log_det(z)
        z = self(z)
        return z, ld


class SimpleStatic(Static):
    def __init__(self, dim):
        super().__init__(dim=dim, h=Identity)

    def inv(self, z1: torch.Tensor):
        u: torch.Tensor = self.u
        w: torch.Tensor = self.w
        b: torch.Tensor = self.b
        h: callable = self.h

        uhat = u + (m(dot(w, u)) - dot(w, u)) * w / dot(w, w)

        bs, d = z1.size()

        I = torch.eye(d).reshape(1, d, d)

        uw_inv = torch.inverse(I + uhat.reshape(bs, d, 1) * w.reshape(bs, 1, d))
        z1bu = (z1 - b * uhat).reshape(bs, 1, d)

        z0 = (uw_inv * z1bu).sum(2)

        return z0


class Dynamic(torch.nn.Module):
    def __init__(self, dim: int, context_size:int, h: DifFunction = Tanh):
        super().__init__()

        self.get_u = torch.nn.Linear(context_size, dim)
        self.get_w = torch.nn.Linear(context_size, dim)
        self.get_b = torch.nn.Linear(context_size, 1)
        self.h = h()

    def get_uwb(self, x: torch.Tensor):
        return self.get_u(x), self.get_w(x), self.get_b(x)

    def forward(self, x: torch.Tensor, v=torch.Tensor):
        u, w, b = self.get_uwb(v)
        y = planar_fn(x=x, w=w, u=u, b=b, h=self.h)
        return y

    def forward_and_logdet(self, x: torch.Tensor, v=torch.Tensor):
        u, w, b = self.get_uwb(v)

        ld = log_det(x=x, w=w, u=u, b=b, h=self.h)
        y = planar_fn(x=x, w=w, u=u, b=b, h=self.h)
        return y, ld
