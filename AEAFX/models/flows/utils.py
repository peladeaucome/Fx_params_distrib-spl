import torch
from ...utils import safe_log
from torch import Tensor
from torch import nn


class DifFunction:
    def __init__(self, f: callable, fp: callable, inv: callable):
        self.f: callable = f
        self.fp: callable = fp
        self.inv: callable = inv

    def __call__(self, x):
        return self.f(x)

    def deriv(self, x):
        return self.fp(x)


class Tanh(DifFunction):
    def __init__(self):
        super().__init__(
            f=torch.nn.functional.tanh,
            fp=lambda x: 1 - torch.square(torch.tanh(x)),
            inv=torch.atanh,
        )


class Identity(DifFunction):
    def __init__(self):
        super().__init__(
            f=lambda x: x,
            fp=lambda x: 1,
            inv=lambda x: x,
        )


class Inverse(DifFunction):
    def __init__(self):
        super().__init__(
            f=lambda x: 1 / x,
            fp=lambda x: -1 / torch.pow(x, 2),
            inv=lambda x: 1 / x,
        )


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def det(self, *z: Tensor):
        z = z[0]
        raise NotImplementedError()

    def log_det(self, *z: Tensor):
        z = z[0]
        return safe_log(self.det(z))

    def forward_and_logdet(self, *z: Tensor):
        z = z[0]
        ld = self.log_det(z)
        z = self(z)
        return z, ld


class PointNonLinearFlow(Flow):
    def __init__(self, f: callable, derivative: callable):
        super().__init__()
        self.f = f
        self.derivative = derivative

    def forward(self, *x: Tensor):
        return self.f(x[0])

    def det(self, *x: Tensor):
        return torch.abs(torch.prod(self.derivative(x[0]), dim=1, keepdim=True))

    def log_det(self, *x: Tensor):
        x = x[0]
        out = torch.sum(safe_log(self.derivative(x).abs()), dim=1, keepdim=True)
        return out

    def forward_and_logdet(self, *z: Tensor):
        z = z[0]
        ld = self.log_det(z)
        z = self(z)
        return z, ld


class SigmoidLayer(PointNonLinearFlow):
    def __init__(self):

        sig = torch.nn.functional.sigmoid
        fp = lambda x: sig(x) * (1 - sig(x))

        super().__init__(
            f=sig,
            derivative=fp,
        )

    def log_det(self, *x: Tensor):
        x = x[0]
        # Better for numerical stability
        out = (-x - 2 * torch.nn.functional.softplus(-x)).sum(dim=1, keepdim=True)
        return out


class ScaledSigmoidLayer(PointNonLinearFlow):
    def __init__(self):

        sig = torch.nn.functional.sigmoid
        f = lambda x: sig(4 * x)
        fp = lambda x: 4 * sig(4 * x) * (1 - sig(4 * x))

        super().__init__(
            f=sig,
            derivative=fp,
        )


class LogitLayer(PointNonLinearFlow):
    def __init__(self):

        logit = torch.logit
        fp = lambda x: 1 / x + 1 / (1 - x)

        super().__init__(f=logit, derivative=fp)


class TanhLayer(PointNonLinearFlow):
    def __init__(self):
        tanh = torch.nn.functional.tanh
        fp = lambda x: 1 - torch.square(tanh(x))
        super().__init__(f=tanh, derivative=fp)


class ELULayer(PointNonLinearFlow):
    def __init__(self):
        elu = lambda x: torch.nn.functional.elu(x, alpha=1)
        fp = lambda x: torch.where(x > 0, torch.ones_like(x), torch.exp(x))
        super().__init__(elu, fp)


class SoftplusLayer(PointNonLinearFlow):
    def __init__(self):
        sp = lambda x: torch.nn.functional.softplus(x)
        fp = lambda x: x.exp() / (1 + x.exp())

        super().__init__(sp, fp)


class LeakyLayer(PointNonLinearFlow):
    def __init__(self, base_layer: PointNonLinearFlow, alpha: float = 0.5):
        f = lambda x: base_layer.f(x) * alpha + (1 - alpha) * x
        fp = lambda x: base_layer.derivative(x) * alpha + (1 - alpha)
        super().__init__(f=f, derivative=fp)


class ParamLayer(PointNonLinearFlow):
    def __init__(self, base_layer: PointNonLinearFlow, dim: int = 1):
        super().__init__(f=base_layer.f, derivative=base_layer.derivative)
        self.alpha = torch.nn.Parameter(torch.zeros(1, dim))

    def forward(self, *x: Tensor):
        a = torch.sigmoid(self.alpha)
        x = x[0]
        return self.f(x) * a + (1 - a) * x

    def det(self, *x_list: Tensor):
        a = torch.sigmoid(self.alpha)
        x = x_list[0]
        out: Tensor = self.derivative(x).abs() * a + (1 - a)
        return out.prod(1, keepdim=True)

    def forward_and_logdet(self, *x: Tensor):
        a = torch.sigmoid(self.alpha)
        x = x[0]
        ld = safe_log(self.derivative(x).abs() * a + (1 - a)).sum(1, keepdim=True)
        x = self.f(x) * a + (1 - a) * x
        return x, ld


class DynamicParamLayer(PointNonLinearFlow):
    def __init__(self, base_layer: PointNonLinearFlow, dim: int, context_size: int):
        super().__init__(f=base_layer.f, derivative=base_layer.derivative)
        self.get_alpha = torch.nn.Sequential(
            torch.nn.Linear(context_size, dim), torch.nn.Sigmoid()
        )

    def forward(self, x: Tensor, c: Tensor):
        alpha = self.get_alpha(c)
        return self.f(x) * alpha + (1 - alpha) * x[0]

    def det(self, x: Tensor, c: Tensor):
        alpha = self.get_alpha(c)
        return torch.abs(
            torch.prod(self.derivative(x) * alpha + (1 - alpha), dim=1, keepdim=True)
        )

    def log_det(self, x: Tensor, c: Tensor):
        return safe_log(self.det(x, c))

    def forward_and_logdet(self, x: Tensor, c: Tensor):
        ld = self.log_det(x, c)
        x = self(x, c)
        return x, ld


class Flow_Sequential(nn.Sequential):
    def det(self, z: Tensor) -> Tensor:
        return torch.exp(self.log_det(z))

    def log_det(self, *args: Tensor) -> Tensor:
        z, c = args
        ld = 0
        for layer in self:
            layer: Flow
            ld = ld + layer.log_det(z, c).mean(1)
        return ld

    def forward(self, *args: Tensor) -> Tensor:
        z, c = args
        for layer in self:
            layer: Flow
            z = layer(z, c)
        return z

    def forward_and_logdet(self, *args: Tensor) -> tuple[Tensor, Tensor]:
        z, c = args
        ld_out = torch.zeros(z.size(0), device=z.device)
        for layer in self:
            layer: Flow
            z, ld = layer.forward_and_logdet(z, c)
            ld_out = ld_out + ld.mean(1)
        return z, ld_out

    def __getitem__(self, idx) -> Flow:
        return super().__getitem__(idx)
