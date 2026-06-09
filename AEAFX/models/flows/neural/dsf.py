import torch
from ..utils import Flow, SigmoidLayer, LogitLayer
from ....utils import safe_log
class SimpleLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class LowTriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            self.weight.copy_(torch.tril(self.weight))
        self.weight.register_hook(lambda grad: grad * torch.tril(torch.ones_like(grad)))



class Static(Flow):
    def __init__(self, dim: int, cond_size: int, hidden_size: int):
        super().__init__()
        self.cond_size = cond_size
        self.get_cond = torch.nn.Sequential(
            LowTriLinear(dim, dim*cond_size),
            torch.nn.ELU()
        )

        self.get_a = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=cond_size, out_channels=hidden_size, kernel_size=1
            ),
            torch.nn.Softplus()
        )
        self.get_b = torch.nn.Conv1d(
            in_channels=cond_size, out_channels=hidden_size, kernel_size=1
        )

        self.get_w = torch.nn.Conv1d(
            in_channels=cond_size, out_channels=hidden_size, kernel_size=1
        )

        self.sigmoid = SigmoidLayer()
        self.logit = LogitLayer()

    def forward(self, x: torch.Tensor):
        bs, dim = x.size()
        cond_size = self.cond_size
        # x = x.unsqueeze(1)

        cond:torch.Tensor = self.get_cond(x)
        cond = cond.view(bs, cond_size, dim)
        # cond.size() = (bs, cond_size)

        a = self.get_a(cond)
        b = self.get_b(cond)
        w = self.get_w(cond)
        w_softmax = torch.softmax(w, axis=1)
        # a.size() = (bs, hidden_size, dim)

        x = torch.sigmoid(x * a + b)

        x = (x * w_softmax).sum(1)

        x = torch.logit(x)
        return x

    def forward_and_logdet(self, x: torch.Tensor):

        bs, in_dim = x.size()
        det = 1


        x = x.unsqueeze(1)

        cond = self.get_cond(x)

        # cond.size() = (bs, cond_size, dim)

        a = self.get_a(cond)
        b = self.get_b(cond)
        w = self.get_w(cond)
        w_softmax = torch.softmax(w, axis=1)
        # a.size() = (bs, hidden_size, dim)

        x, d = self.sigmoid.forward_and_logdet(x * a + b)

        
        det = torch.exp(d)
        det = det*a*w_softmax.sum(1)
        log_det = safe_log(det)

        S = (x * w_softmax).sum(1)

        log_det = log_det+ safe_log(w_softmax)

        x , ld= self.logit.forward_and_logdet(S)

        log_det += ld
        log_det = torch.sum(log_det, dim=1, keepdim=True)
        return x, log_det