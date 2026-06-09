import torch

from ..utils import Flow, PointNonLinearFlow


class Static(Flow):
    def __init__(self, dim: int, num_knots: int = 5):
        super().__init__()
        self.dim = dim
        self.num_knots = num_knots
        self.widths = torch.nn.Parameter(torch.Tensor(1, dim, num_knots - 1))
        self.heights = torch.nn.Parameter(torch.Tensor(1, dim, num_knots - 1))
        self.derivatives = torch.nn.Parameter(torch.Tensor(1, dim, num_knots))

    def forward(self, z: torch.Tensor):
        bs, dim = z.size()
        z=z.unsqueeze(2)

        widths = torch.softmax(self.widths, dim=2)
        heights = torch.softmax(self.heights, dim=2)
        d = torch.nn.functional.softplus(self.derivatives)

        x = torch.zeros(1, dim, self.num_knots, device=z.device)
        x[:, :, 1:] = torch.cumsum(widths, dim=2)

        y = torch.zeros(1, dim, self.num_knots, device=z.device)
        y[:, :, 1:] = torch.cumsum(heights, dim=2)

        s = (y[:, :, 1:] - y[:, :, :-1]) / widths

        zeta = z - x

        alpha_0 = y[:, :, :-1]
        alpha1 = d[:, :, :-1]
        alpha2 = (3 * s - 2 * d[:, :, :-1] - d[:, :, 1:]) / widths
        alpha2 = (d[:, :, :-1] + d[:, :, 1:] - 2 * s) / widths.square()

        idx = torch.searchsorted(x, z)

        return z


class Dynamic(Flow):
    def __init__(self, dim, context_size, num_knots):
        self.get_widths = 1
