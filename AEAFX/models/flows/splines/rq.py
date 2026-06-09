import torch
from torch import Tensor
import torch.nn as nn
from ..utils import Flow
from ....utils import safe_log
from ...layers import Swish, ResBlock


class Dynamic(Flow):
    def __init__(
        self,
        dim: int,
        context_size: int,
        num_knots: int,
        bound: float,
        min_val: float = 0.01,
    ):
        super().__init__()
        self.num_knots = num_knots
        self.bound = bound
        self.dim = dim

        self.get_w = nn.Sequential(
            nn.Linear(context_size, (num_knots - 1) * dim),
            nn.Unflatten(dim=1, unflattened_size=(dim, (num_knots - 1))),
            SoftplusMax(dim=2),
        )
        self.get_h = nn.Sequential(
            nn.Linear(context_size, (num_knots - 1) * dim),
            nn.Unflatten(dim=1, unflattened_size=(dim, (num_knots - 1))),
            SoftplusMax(dim=2),
        )
        self.get_d = nn.Sequential(
            nn.Linear(context_size, (num_knots - 2) * dim),
            nn.Unflatten(dim=1, unflattened_size=(dim, (num_knots - 2))),
            nn.Softplus(),
        )

        self.get_w[0].weight.data.fill_(0)
        self.get_w[0].bias.data.fill_(0)
        self.get_h[0].weight.data.fill_(0)
        self.get_h[0].bias.data.fill_(0)
        self.get_d[0].weight.data.fill_(0)
        self.get_d[0].bias.data.fill_(1)

        self.min_val = min_val

    def forward(self, z: Tensor, c: Tensor):
        w = self.get_w(c) * 2 * self.bound
        h = self.get_h(c) * 2 * self.bound
        d_small = self.get_d(c) + self.min_val

        out = compute_spline(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        return out

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(c) * 2 * self.bound
        h = self.get_h(c) * 2 * self.bound
        d_small = self.get_d(c) + self.min_val

        out, deriv = compute_spline_and_deriv(
            z=z, h=h, w=w, d_small=d_small, bound=self.bound
        )
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return out, logdet

    def logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(c) * 2 * self.bound
        h = self.get_h(c) * 2 * self.bound
        d_small = self.get_d(c) + self.min_val

        deriv = compute_deriv(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return logdet


class Coupling(Flow):
    def __init__(
        self,
        dim: int,
        context_size: int,
        num_knots: int,
        bound: float,
        min_val: float = 0.01,
    ):
        """
        Inputs:
        -------
        `dim` : int
            Dimension of the flow inputs
        `context_size`: int
            Dimension of the context vector
        `num_knots`: int
            Number of knots of the spline. This number includes the two outer knots.
        `bound`: float
            Bounds of the spline
        `min_val`: float
            minimum value to help stability
        """
        super().__init__()
        self.num_knots = num_knots
        self.bound = bound
        self.dim = dim

        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        bn = True

        self.get_w1 = nn.Sequential(
            # nn.Linear(context_size, context_size),
            # Swish(context_size),
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 1) * self.dim1),
            nn.Unflatten(dim=1, unflattened_size=(self.dim1, (num_knots - 1))),
            nn.Softplus(),
        )
        self.get_w2 = nn.Sequential(
            # nn.Linear(context_size + self.dim1, context_size + self.dim1),
            # Swish(context_size + self.dim1),
            ResBlock(context_size + self.dim1, bn=bn),
            nn.Linear(context_size + self.dim1, (num_knots - 1) * self.dim2),
            nn.Unflatten(dim=1, unflattened_size=(self.dim2, (num_knots - 1))),
            nn.Softplus(),
        )

        self.get_h1 = nn.Sequential(
            # nn.Linear(context_size, context_size),
            # Swish(context_size),
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 1) * self.dim1),
            nn.Unflatten(dim=1, unflattened_size=(self.dim1, (num_knots - 1))),
            nn.Softplus(),
        )
        self.get_h2 = nn.Sequential(
            # nn.Linear(context_size + self.dim1, context_size + self.dim1),
            # Swish(context_size + self.dim1),
            ResBlock(context_size + self.dim1, bn=bn),
            nn.Linear(context_size + self.dim1, (num_knots - 1) * self.dim2),
            nn.Unflatten(dim=1, unflattened_size=(self.dim2, (num_knots - 1))),
            nn.Softplus(),
        )

        self.get_d1 = nn.Sequential(
            # nn.Linear(context_size, context_size),
            # Swish(context_size),
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 2) * self.dim1),
            nn.Unflatten(dim=1, unflattened_size=(self.dim1, (num_knots - 2))),
            nn.Softplus(),
        )
        self.get_d2 = nn.Sequential(
            # nn.Linear(context_size + self.dim1, context_size + self.dim1),
            # Swish(context_size + self.dim1),
            ResBlock(context_size + self.dim1, bn=bn),
            nn.Linear(context_size + self.dim1, (num_knots - 2) * self.dim2),
            nn.Unflatten(dim=1, unflattened_size=(self.dim2, (num_knots - 2))),
            nn.Softplus(),
        )

        self.min_val = min_val

        # self.get_w1[0].weight.data.fill_(0)
        # self.get_w1[0].bias.data.fill_(1)
        # self.get_h1[0].weight.data.fill_(0)
        # self.get_h1[0].bias.data.fill_(1)
        # self.get_d1[0].weight.data.fill_(0)
        # self.get_d1[0].bias.data.fill_(1)
        # self.get_w2[0].weight.data.fill_(0)
        # self.get_w2[0].bias.data.fill_(1)
        # self.get_h2[0].weight.data.fill_(0)
        # self.get_h2[0].bias.data.fill_(1)
        # self.get_d2[0].weight.data.fill_(0)
        # self.get_d2[0].bias.data.fill_(1)

    def get_w(self, z, c):
        dim = self.dim
        bs = z.size(0)

        z1, z2 = torch.split(z, [self.dim1, self.dim2], 1)

        w1 = self.get_w1(c)
        w2 = self.get_w2(torch.cat((c, z1), dim=1))
        w = torch.cat((w1, w2), dim=1)
        w = w / torch.sum(w, dim=2, keepdim=True)
        w = w + 1e-2
        w = w / torch.sum(w, dim=2, keepdim=True)
        return w

    def get_h(self, z, c):
        dim = self.dim
        bs = z.size(0)

        z1, z2 = torch.split(z, [self.dim1, self.dim2], 1)

        h1 = self.get_h1(c)
        h2 = self.get_h2(torch.cat((c, z1), dim=1))
        h = torch.cat((h1, h2), dim=1)
        h = h / torch.sum(h, dim=2, keepdim=True)
        h = h + 1e-2
        h = h / torch.sum(h, dim=2, keepdim=True)
        return h

    def get_d(self, z, c):
        dim = self.dim
        bs = z.size(0)

        z1, z2 = torch.split(z, [self.dim1, self.dim2], 1)

        d1 = self.get_d1(c)
        d2 = self.get_d2(torch.cat((c, z1), dim=1))
        d = torch.cat((d1, d2), dim=1) + self.min_val
        return d

    def forward(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        out = compute_spline(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        return out

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        out, deriv = compute_spline_and_deriv(
            z=z, h=h, w=w, d_small=d_small, bound=self.bound
        )
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return out, logdet

    def logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        deriv = compute_deriv(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return logdet


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
    def __init__(self, dim: int):
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
        self.bias = nn.Parameter(torch.randn(1, dim, num_chans))

    def forward(self, x: Tensor):
        x = x.unsqueeze(2)
        return x * self.weight + self.bias


class Dynamic_AR(Flow):
    def __init__(
        self,
        dim: int,
        context_size: int,
        num_knots: int,
        bound: float,
        min_val: float = 0.01,
    ):
        """
        Inputs:
        -------
        `dim` : int
            Dimension of the flow inputs
        `context_size`: int
            Dimension of the context vector
        `num_knots`: int
            Number of knots of the spline. This number includes the two outer knots.
        `bound`: float
            Bounds of the spline
        `min_val`: float
            minimum value to help stability
        """
        super().__init__()
        self.num_knots = num_knots
        self.bound = bound
        self.dim = dim

        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        bn = True

        self.get_w_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 1) * self.dim),
            nn.Unflatten(dim=1, unflattened_size=(self.dim, (num_knots - 1))),
            nn.Softplus(),
        )
        self.get_w_from_z = nn.Sequential(
            TriangleFCBlock(dim=dim),
            Adaptor(dim=dim, num_chans=num_knots - 1),
            nn.Softplus(),
        )

        self.get_h_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 1) * self.dim),
            nn.Unflatten(dim=1, unflattened_size=(self.dim, (num_knots - 1))),
            nn.Softplus(),
        )
        self.get_h_from_z = nn.Sequential(
            TriangleFCBlock(dim=dim),
            Adaptor(dim=dim, num_chans=num_knots - 1),
            nn.Softplus(),
        )

        self.get_d_from_c = nn.Sequential(
            ResBlock(context_size, bn=bn),
            nn.Linear(context_size, (num_knots - 2) * self.dim),
            nn.Unflatten(dim=1, unflattened_size=(self.dim, (num_knots - 2))),
            nn.Softplus(),
        )
        self.get_d_from_z = nn.Sequential(
            TriangleFCBlock(dim=dim),
            Adaptor(dim=dim, num_chans=num_knots - 2),
            nn.Softplus(),
        )

        self.min_val = min_val

        # self.get_w1[0].weight.data.fill_(0)
        # self.get_w1[0].bias.data.fill_(1)
        # self.get_h1[0].weight.data.fill_(0)
        # self.get_h1[0].bias.data.fill_(1)
        # self.get_d1[0].weight.data.fill_(0)
        # self.get_d1[0].bias.data.fill_(1)
        # self.get_w2[0].weight.data.fill_(0)
        # self.get_w2[0].bias.data.fill_(1)
        # self.get_h2[0].weight.data.fill_(0)
        # self.get_h2[0].bias.data.fill_(1)
        # self.get_d2[0].weight.data.fill_(0)
        # self.get_d2[0].bias.data.fill_(1)

    def get_w(self, z, c):
        dim = self.dim
        bs = z.size(0)

        w = self.get_w_from_z(z) + self.get_w_from_c(c)

        w = w / torch.sum(w, dim=2, keepdim=True)
        w = w + 1e-2
        w = w / torch.sum(w, dim=2, keepdim=True)
        return w

    def get_h(self, z, c):
        dim = self.dim
        bs = z.size(0)

        h = self.get_h_from_z(z) + self.get_h_from_c(c)

        h = h / torch.sum(h, dim=2, keepdim=True)
        h = h + 1e-2
        h = h / torch.sum(h, dim=2, keepdim=True)
        return h

    def get_d(self, z, c):
        dim = self.dim
        bs = z.size(0)

        d = self.get_d_from_z(z) + self.get_d_from_c(c)

        d = d + self.min_val
        return d

    def forward(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        out = compute_spline(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        return out

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        out, deriv = compute_spline_and_deriv(
            z=z, h=h, w=w, d_small=d_small, bound=self.bound
        )
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return out, logdet

    def logdet(self, z: Tensor, c: Tensor):
        w = self.get_w(z, c) * 2 * self.bound
        h = self.get_h(z, c) * 2 * self.bound
        d_small = self.get_d(z, c)

        deriv = compute_deriv(z=z, h=h, w=w, d_small=d_small, bound=self.bound)
        logdet = safe_log(deriv).sum(1, keepdim=True)

        return logdet


def compute_spline(
    z: Tensor,
    h: Tensor,
    w: Tensor,
    d_small: Tensor,
    bound: float,
) -> Tensor:

    bs, dim = z.size()
    z = z.unsqueeze(2)

    num_knots = h.size(2) + 1

    d = torch.ones(bs, dim, num_knots, device=z.device)
    d[:, :, 1:-1] = d_small

    x = torch.zeros(bs, dim, num_knots, device=z.device)
    x[:, :, 1:] = torch.cumsum(w, dim=2) - bound
    x[:, :, 0] = -bound
    x[:, :, -1] = bound

    y = torch.zeros(bs, dim, num_knots, device=z.device)
    y[:, :, 1:] = torch.cumsum(h, dim=2) - bound
    y[:, :, 0] = -bound
    y[:, :, -1] = bound

    s = h / w
    zeta = (z - x[:, :, :-1]) / w

    out = (h) * (s * zeta.square() + d[:, :, :-1] * zeta * (1 - zeta))
    out = out / (s + (d[:, :, 1:] + d[:, :, :-1] - 2 * s) * zeta * (1 - zeta))
    out = out + y[:, :, :-1]

    idx = torch.searchsorted(x.expand(bs, dim, num_knots), z) - 1
    idx = torch.maximum(idx, torch.zeros_like(idx))
    idx = torch.minimum(idx, torch.ones_like(idx) * (num_knots - 2))

    out = torch.gather(out, 2, idx)
    out = torch.where(z > bound, z, out)
    out = torch.where(z < -bound, z, out)

    out = out.squeeze(2)

    return out


def compute_deriv(
    z: Tensor,
    h: Tensor,
    w: Tensor,
    d_small: Tensor,
    bound: float,
) -> Tensor:
    bs, dim = z.size()
    z = z.unsqueeze(2)

    num_knots = h.size(2) + 1

    d = torch.ones(bs, dim, num_knots, device=z.device)
    d[:, :, 1:-1] = d_small

    x = torch.zeros(bs, dim, num_knots, device=z.device)
    x[:, :, 1:] = torch.cumsum(w, dim=2) - bound
    x[:, :, 0] = -bound
    x[:, :, -1] = bound

    y = torch.zeros(bs, dim, num_knots, device=z.device)
    y[:, :, 1:] = torch.cumsum(h, dim=2) - bound
    y[:, :, 0] = -bound
    y[:, :, -1] = bound

    s = h / w
    zeta = (z - x[:, :, :-1]) / w

    deriv = s.square() * (
        d[:, :, 1:] * zeta.square()
        + 2 * s * zeta * (1 - zeta)
        + d[:, :, :-1] * ((1 - zeta).square())
    )
    deriv = deriv / (
        ((s + (d[:, :, 1:] + d[:, :, :-1] - 2 * s) * zeta * (1 - zeta))).square()
    )

    idx = torch.searchsorted(x.expand(bs, dim, num_knots), z) - 1
    idx = torch.maximum(idx, torch.zeros_like(idx))
    idx = torch.minimum(idx, torch.ones_like(idx) * (num_knots - 2))

    deriv = torch.gather(deriv, 2, idx)
    ones = torch.ones_like(deriv)
    deriv = torch.where(z > bound, ones, deriv)
    deriv = torch.where(z < -bound, ones, deriv)

    deriv = deriv.squeeze(2)

    return deriv


def compute_spline_and_deriv(
    z: Tensor,
    h: Tensor,
    w: Tensor,
    d_small: Tensor,
    bound: float,
) -> tuple[Tensor]:

    bs, dim = z.size()
    z = z.unsqueeze(2)

    num_knots = h.size(2) + 1

    d = torch.ones(bs, dim, num_knots, device=z.device)
    d[:, :, 1:-1] = d_small

    x = torch.zeros(bs, dim, num_knots, device=z.device)
    x[:, :, 1:] = torch.cumsum(w, dim=2) - bound
    x[:, :, 0] = -bound
    x[:, :, -1] = bound

    y = torch.zeros(bs, dim, num_knots, device=z.device)
    y[:, :, 1:] = torch.cumsum(h, dim=2) - bound
    y[:, :, 0] = -bound
    y[:, :, -1] = bound

    s = h / w
    zeta = (z - x[:, :, :-1]) / w

    out = (h) * (s * zeta.square() + d[:, :, :-1] * zeta * (1 - zeta))
    out = out / (s + (d[:, :, 1:] + d[:, :, :-1] - 2 * s) * zeta * (1 - zeta))
    out = out + y[:, :, :-1]

    deriv = s.square() * (
        d[:, :, 1:] * zeta.square()
        + 2 * s * zeta * (1 - zeta)
        + d[:, :, :-1] * ((1 - zeta).square())
    )
    deriv = deriv / (
        ((s + (d[:, :, 1:] + d[:, :, :-1] - 2 * s) * zeta * (1 - zeta))).square()
    )

    idx = torch.searchsorted(x.expand(bs, dim, num_knots), z) - 1
    idx = torch.maximum(idx, torch.zeros_like(idx))
    idx = torch.minimum(idx, torch.ones_like(idx) * (num_knots - 2))

    out = torch.gather(out, 2, idx)
    out = torch.where(z > bound, z, out)
    out = torch.where(z < -bound, z, out)

    deriv = torch.gather(deriv, 2, idx)
    ones = torch.ones_like(deriv)
    deriv = torch.where(z > bound, ones, deriv)
    deriv = torch.where(z < -bound, ones, deriv)

    out = out.squeeze(2)
    deriv = deriv.squeeze(2)

    return out, deriv


class SoftplusMax(nn.Softplus):
    def __init__(self, dim: int = 1, beta: int = 1, threshold: int = 20) -> None:
        super().__init__(beta, threshold)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        out = out / out.sum(self.dim, keepdim=True)
        return out


class StableSoftplusMax(SoftplusMax):
    def __init__(
        self, eps: int = 1e-2, dim: int = 1, beta: int = 1, threshold: int = 20
    ) -> None:
        super().__init__(beta, threshold)
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        out = out / out.sum(self.dim, keepdim=True)
        out = out + self.eps
        out = out / out.sum(self.dim, keepdim=True)
        return out
