import torch
import torch.nn as nn
from torch import Tensor
from .utils import Flow
from ...utils import safe_log
from ..layers import Swish


class LoTriLinear(nn.Linear):
    def __init__(self, in_features: int, offset: int = 0):
        super().__init__(in_features, in_features, bias=False)
        with torch.no_grad():
            self.weight.data.copy_(torch.eye(in_features))
            self.weight.copy_(torch.tril(self.weight))
        self.weight.register_hook(
            lambda grad: grad * torch.tril(torch.ones_like(grad), offset)
        )


class UpTriLinear(nn.Linear):
    def __init__(self, in_features, offset: int = 0):
        super().__init__(in_features, in_features, bias=False)
        with torch.no_grad():
            self.weight.data.copy_(torch.eye(in_features))
            self.weight.copy_(torch.triu(self.weight))
        self.weight.register_hook(
            lambda grad: grad * torch.triu(torch.ones_like(grad), offset)
        )


class StaticLower(Flow):
    def __init__(self, dim: int, offset: int = 0):
        super().__init__()
        self.dim = dim
        self.offset = offset
        self.lin = LoTriLinear(dim, offset=offset)

    def forward(self, *z_tuple):
        z = z_tuple[0]
        z = self.lin(z)
        return z

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = self.lin(z)

        log_det = (
            torch.log(torch.diag(self.lin.weight).abs())
            .sum(0, keepdim=True)
            .unsqueeze(0)
        )

        return z, log_det


class Rectangular(Flow):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)

    def forward(self, *z_tuple):
        z = z_tuple[0]
        z = self.lin(z)
        return z

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = self.lin(z)

        log_det = safe_log(torch.matmul(self.lin.weight, self.lin.weight.T).det().abs())
        log_det = log_det.view(1, 1) / 2

        return z, log_det


class StaticUpper(Flow):
    def __init__(self, dim: int, offset: int = 0):
        super().__init__()
        self.dim = dim

        self.lin = UpTriLinear(dim, offset=offset)

    def forward(self, *z_tuple):
        z = z_tuple[0]
        z = self.lin(z)
        return z

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = self.lin(z)

        log_det = (
            torch.log(torch.diag(self.lin.weight).abs())
            .sum(0, keepdim=True)
            .unsqueeze(0)
        )

        return z, log_det


class DynamicLower(Flow):
    def __init__(self, dim: int, context_size: int, offset: int = 0):
        super().__init__()
        self.dim = dim
        self.num_params = int((dim + offset) * ((dim + offset) + 1) / 2)
        self.offset = offset

        # self.get_weights = nn.Linear(context_size, self.num_params)
        self.get_weights = nn.Sequential(
            nn.Linear(context_size, context_size),
            Swish(context_size),
            nn.Linear(context_size, self.num_params),
        )
        # self.get_biases = nn.Linear(context_size, dim)

        # self.get_weights.weight.data.fill_(0.0)
        # self.get_weights.bias.data.fill_(0.0)

    def get_matrix(self, c: Tensor):
        bs = c.size(0)
        dim = self.dim
        mat = torch.zeros(bs, dim, dim, device=c.device)
        weights_vec = self.get_weights(c)

        # weights_vec = torch.tanh(weights_vec)

        idx = torch.tril_indices(dim, dim, offset=0).tolist()
        mat[:, idx[0], idx[1]] = weights_vec

        mat = mat * 0.1 + torch.eye(dim, device=mat.device).unsqueeze(0)
        return mat

    @staticmethod
    def matmul(mat: Tensor, vec: Tensor):
        vec = vec.unsqueeze(2)

        # out = (vec * mat).sum(2)
        out = torch.matmul(mat, vec).squeeze(2)
        # print(mat.size(), vec.size(), out.size())
        return out

    def forward(self, z: Tensor, c: Tensor):
        mat = self.get_matrix(c)

        z = self.matmul(mat, z)
        return z

    def det(self, z, c):
        mat = self.get_matrix(c)

        z = self.matmul(mat, z)
        det = torch.prod(
            torch.diagonal(mat, offset=0, dim1=1, dim2=2), dim=1, keepdim=True
        )
        return det

    def logdet(self, z, c):
        mat = self.get_matrix(c)

        z = self.matmul(mat, z)
        det = torch.prod(
            torch.diagonal(mat, offset=0, dim1=1, dim2=2), dim=1, keepdim=True
        )
        logdet = safe_log(torch.abs(det))
        return logdet

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        mat = self.get_matrix(c)

        z = self.matmul(mat, z)
        logdet = safe_log(torch.diagonal(mat, offset=0, dim1=1, dim2=2).abs()).sum(
            1, keepdim=True
        )
        return z, logdet


class DynamicUpper(DynamicLower):
    def get_matrix(self, c: Tensor):
        return super().get_matrix(c).transpose(1, 2)


class StaticBias(Flow):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(1, dim))

    def forward(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = z + self.bias
        return z

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = z + self.bias

        bs = z.size(0)
        logdet = torch.zeros(bs, 1, device=z.device)
        return z, logdet


class DynamicBias(Flow):
    def __init__(self, dim: int, context_size: int):
        super().__init__()
        self.dim = dim

        self.get_biases = nn.Linear(context_size, dim)

        self.get_biases.weight.data.fill_(0.0)
        self.get_biases.bias.data.fill_(0.0)

    def forward(self, z: Tensor, c: Tensor):

        b = self.get_biases(c)
        z = z + b
        return z

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        bs = z.size(0)
        b = self.get_biases(c)
        z = z + b

        logdet = torch.zeros(bs, 1, device=z.device)
        return z, logdet


class CouplingBias(Flow):
    def __init__(self, dim: int, context_size: int):
        super().__init__()
        self.dim = dim
        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        self.get_biases1 = nn.Linear(context_size, self.dim1)
        self.get_biases2 = nn.Linear(context_size + self.dim1, self.dim2)

        self.get_biases1.weight.data.fill_(0.0)
        self.get_biases1.bias.data.fill_(0.0)
        self.get_biases2.weight.data.fill_(0.0)
        self.get_biases2.bias.data.fill_(0.0)

    def get_biases(self, z: Tensor, c: Tensor):
        dim = self.dim
        bs = z.size(0)

        z1, z2 = torch.split(z, [self.dim1, self.dim2], 1)

        b1 = self.get_biases1(c)
        b2 = self.get_biases2(torch.cat((c, z1), dim=1))
        b = torch.cat((b1, b2), dim=1)
        return b

    def forward(self, z: Tensor, c: Tensor):
        b = self.get_biases(z, c)

        z = z + b
        return z

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        bs = z.size(0)
        b = self.get_biases(z, c)
        z = z + b

        logdet = torch.zeros(bs, 1, device=z.device)
        return z, logdet


class StaticDiag(Flow):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.ones(1, dim))

    def forward(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = z * self.weights
        return z

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        z = z * self.weights

        logdet = safe_log(self.weights).sum(dim=1, keepdim=True)
        return z, logdet


class DynamicDiag(Flow):
    def __init__(self, dim: int, context_size: int):
        super().__init__()
        self.dim = dim
        self.get_weights = nn.Linear(context_size, dim)

        self.get_weights.weight.data.fill_(0.0)
        self.get_weights.bias.data.fill_(1.0)

    def forward(self, z: Tensor, c: Tensor):
        w:Tensor = self.get_weights(c)
        z = z * w.exp()
        return z

    def forward_and_logdet(self, z: Tensor, c: Tensor):
        w: Tensor = self.get_weights(c)
        z = z * w.exp()
        logdet = w.sum(dim=1, keepdim=True)
        return z, logdet


class RandomPermutation(Flow):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # self.permutations = torch.randperm(n=dim).tolist()
        self.permutations = nn.Parameter(
            torch.randperm(n=dim).unsqueeze(0), requires_grad=False
        )
        # self.register_buffer("perm", self.permutations)

    def forward(self, *z_tuple: Tensor):
        z = z_tuple[0]
        bs = z.size(0)

        perm = self.permutations.expand(bs, self.dim)

        out = torch.gather(z, 1, perm)
        return out

    def det(self, *z_tuple: Tensor):
        z = z_tuple[0]
        return torch.ones(z.size(0), 1, device=z.device)

    def logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        return torch.zeros(z.size(0), 1, device=z.device)

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        bs = z.size(0)

        perm = self.permutations.expand(bs, self.dim)

        out = torch.gather(z, 1, perm)
        return out, torch.zeros(z.size(0), 1, device=z.device)

class InvertPermutation(Flow):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *z_tuple: Tensor):
        z = z_tuple[0]
        out = torch.flip(z, dims=[1])
        return out

    def det(self, *z_tuple: Tensor):
        z = z_tuple[0]
        return torch.ones(z.size(0), 1, device=z.device)

    def logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        return torch.zeros(z.size(0), 1, device=z.device)

    def forward_and_logdet(self, *z_tuple: Tensor):
        z = z_tuple[0]
        bs = z.size(0)

        out = torch.flip(z, dims=[1])

        return out, torch.zeros(z.size(0), 1, device=z.device)