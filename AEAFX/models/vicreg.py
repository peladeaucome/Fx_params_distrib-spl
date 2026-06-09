import torch.utils
import torch.utils.data
from . import frontend
import torch
import torch.nn as nn
import lightning.pytorch as pl
from ..ddafx import DDAFX
from ..utils import safe_log
from typing import Optional, Union, Literal, LiteralString, Callable
from ..loss import abs_params_loss
from .distributions import (
    Distribution,
    Dirac,
    Gaussian,
    Normal,
    Uniform,
    GMMUniform,
    GMMFull,
)
from torch import Tensor
from .layers import Swish, ResBlock, FCBlock


# from .flow_layers import PlanarLayer, SimplePlanarLayer, SigmoidLayer, SigmoidLayer

from . import flows
import numpy as np



from .main import FX_AE, FX_Inference


class FX_AE_VICReg(FX_AE):
    def __init__(
        self,
        frontend_args: dict[str, str],
        ana_fx: DDAFX,
        syn_fx: DDAFX,
        mlp_depth: tuple[int] = 2,
        mlp_size: int = 64,
        mlp_type: Literal["mlp", "res"] = "mlp",
        dropout: float = 0.0,
        loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: dict = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        vicreg_weight: float = 1,
    ):
        super().__init__(
            frontend_args=frontend_args,
            fx=ana_fx,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            dropout=dropout,
            metrics_dict=metrics_dict,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
        )

        self.syn_fx = syn_fx
        self.vicreg_weight = vicreg_weight

    def init_mlp(self):
        self.mlp = nn.Sequential()
        if self.frontend.out_dim != self.mlp_size:
            self.mlp.append(
                FCBlock(self.frontend.out_dim, self.mlp_size, dropout=self.dropout),
            )

        for idx in range(self.mlp_depth):
            if self.mlp_type == "mlp":
                self.mlp.append(
                    FCBlock(self.mlp_size, self.mlp_size, dropout=self.dropout)
                )
            elif self.mlp_type == "res":
                self.mlp.append(ResBlock(dim=self.mlp_size, dropout=self.dropout))

        self.mlp.append(nn.Linear(self.mlp_size, self.fx.num_parameters))
        # self.mlp.append(nn.BatchNorm1d(num_features=self.fx.num_parameters))
        self.mlp.append(nn.Sigmoid())

    def get_FXParams(self, x: Tensor, y: Tensor) -> Tensor:

        e = self.frontend(y)

        v_hat = self.mlp(e)
        return v_hat, e

    def forward(self, x: Tensor, y: Tensor):
        z = self.get_FXParams(x, y)
        yh = self.fx(x, z)
        return yh

    def training_step(self, x: Tensor, batch_idx: int):
        bs = x.size(0)

        v = torch.rand((4, self.syn_fx.num_parameters), device=x.device)
        v = v.unsqueeze(1).expand(4, bs // 4, self.syn_fx.num_parameters)
        v = v.flatten(0, 1)

        y = self.syn_fx(x, v)

        z, embedding = self.get_FXParams(x, y)
        y_hat = self.fx(x, z)

        e = embedding.unflatten(0, (4, bs // 4))

        same_var = e.var(1).mean()
        opposite_var = e.var(0).mean()

        ssl_loss = same_var - torch.tanh(e.var(0)).mean()

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()
        total_loss = loss_audio + ssl_loss * self.vicreg_weight

        self.log("loss_audio/train", loss_audio)
        self.log("loss_total/train", total_loss)
        self.log("loss_params/train", loss_params)
        self.log("ssl_loss/train", ssl_loss)
        self.log("same_var/train", same_var)
        self.log("opposite_var/train", opposite_var)

        return total_loss

    def validation_step(self, x: Tensor, batch_idx: int):
        bs = x.size(0)
        v = torch.rand((bs, self.syn_fx.num_parameters), device=x.device)
        
        y = self.syn_fx(x, v)

        z, embedding = self.get_FXParams(x, y)
        y_hat = self.fx(x, z)

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()

        embedding_var = embedding.var(0).mean()

        self.log("loss_audio/valid", loss_audio)
        self.log("loss_total/valid", loss_audio)
        self.log("loss_params/valid", loss_params)
        self.log("embedding_var/valid", embedding_var)
        self.log("params_std/valid", z.std(0).mean(0))
        return loss_audio

    def test_step(self, x: Tensor, batch_idx: int):
        bs = x.size(0)
        v = torch.rand((bs, self.syn_fx.num_parameters), device=x.device)
        y = self.syn_fx(x, v)

        z, embedding = self.get_FXParams(x, y)
        y_hat = self.fx(x, z)

        loss_params = self.param_loss_fn(v, z).mean()

        self.test_procedure(target=y, estimate=y_hat, name="Estimated")
        self.test_procedure(target=y, estimate=x, name="Input")
        self.log("Test/Estimated/Params", loss_params)

    def to(self, device: torch.device):
        self.syn_fx.to(device)
        super().to(device)
        return self

    def cpu(self):
        self.syn_fx.cpu()
        super().cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.syn_fx.cuda(device)
        super().cuda(device)
        return self


class FX_Inference_VicReg(FX_Inference):
    def __init__(
        self,
        ana_fx: DDAFX,
        syn_fx: DDAFX,
        frontend_args: dict[str, str],
        mlp_depth: int,
        mlp_size: int,
        mlp_type: Literal["mlp", "res"] = "mlp",
        start_beta=1,
        end_beta=1,
        warmup_length: int = 15,
        audio_loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: Callable[[], dict] = None,
        flow_length: int = 5,
        flow_layers_type: str = "static",
        flow_nl: str = "res",
        flow_nl_knots: int = 4,
        flow_coupling: bool = False,
        context_size: int = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        base_distrib_str: Literal[
            "gaussian", "gmm_full", "normal", "uniform", "gmm_uniform"
        ] = "gaussian",
        num_mixtures: int = 10,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(
            fx=ana_fx,
            frontend_args=frontend_args,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            start_beta=start_beta,
            end_beta=end_beta,
            warmup_length=warmup_length,
            audio_loss_fn=audio_loss_fn,
            flow_length=flow_length,
            flow_layers_type=flow_layers_type,
            flow_nl=flow_nl,
            flow_nl_knots=flow_nl_knots,
            flow_coupling=flow_coupling,
            context_size=context_size,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            base_distrib_str=base_distrib_str,
            num_mixtures=num_mixtures,
            base_entropy=base_entropy,
            metrics_dict=metrics_dict,
            weight_decay=weight_decay,
        )
        self.syn_fx = syn_fx

    def training_step(self, x: Tensor, batch_idx: int):
        ramp_ratio = min(self.trainer.current_epoch / self.warmup_length, 1)
        self.beta = np.exp(
            np.log(self.start_beta) * (1 - ramp_ratio)
            + np.log(self.end_beta) * ramp_ratio
        )
        # self.beta = self.start_beta * (1 - ramp_ratio) + self.end_beta * ramp_ratio

        bs = x.size(0)
        # v = torch.rand((1, self.syn_fx.num_parameters), device=x.device).expand(
        #     bs, self.syn_fx.num_parameters
        # )

        v = torch.rand((4, self.syn_fx.num_parameters), device=x.device)
        v = v.unsqueeze(1).expand(4, bs // 4, self.syn_fx.num_parameters)
        v = v.flatten(0, 1)

        y = self.syn_fx(x, v)

        audio_loss, neg_entropy, params_loss, embedding = self.compute_losses_optim(
            x, y, v
        )

        e = embedding.unflatten(0, (4, bs // 4))

        same_var = e.var(1).mean()
        opposite_var = e.var(0).mean()

        ssl_loss = same_var - torch.tanh(e.var(0)).mean()

        total_loss = audio_loss + self.beta * neg_entropy + ssl_loss * 1

        entropy_bits = (
            -neg_entropy
            * (torch.ones(1, device=neg_entropy.device).exp().log2())
            * self.params_dim
        )

        self.log("loss_audio/train", audio_loss)
        self.log("loss_params/train", params_loss)
        self.log("loss_total/train", total_loss)
        self.log("entropy_bits/train", entropy_bits)
        self.log("entropy_bits_per_dim/train", entropy_bits / self.params_dim)
        self.log("ssl_loss/train", ssl_loss)
        self.log("same_var/train", same_var)
        self.log("opposite_var/train", opposite_var)
        self.log("beta", self.beta)
        return total_loss

    def validation_step(self, x: Tensor, batch_idx: int):
        bs = x.size(0)
        v = torch.rand((bs, self.syn_fx.num_parameters), device=x.device)

        # v = torch.rand((2, self.syn_fx.num_parameters), device=x.device)
        # v = v.unsqueeze(1).expand(2, bs//2, self.syn_fx.num_parameters)
        # v = v.flatten(0, 1)

        y = self.syn_fx(x, v)

        audio_loss, neg_entropy, params_loss, embedding = self.compute_losses_optim(
            x, y, v
        )

        # e = embedding.unflatten(0, (2, bs//2))

        embedding_var = embedding.var(0).mean()

        total_loss = audio_loss + self.beta * neg_entropy

        entropy_bits = (
            -neg_entropy
            * (torch.ones(1, device=neg_entropy.device).exp().log2())
            * self.params_dim
        )

        self.log("loss_audio/valid", audio_loss)
        self.log("loss_params/valid", params_loss)
        self.log("loss_total/valid", total_loss)
        self.log("entropy_bits/valid", entropy_bits)
        self.log("entropy_bits_per_dim/valid", entropy_bits / self.params_dim)
        self.log("embedding_var/valid", embedding_var)
        self.log("beta", self.beta)
        return total_loss

    def test_step(self, x: Tensor, batch_idx: int):
        bs = x.size(0)
        v = torch.rand((bs, self.syn_fx.num_parameters), device=x.device)
        y = self.syn_fx(x, v)

        v_hat_1, H_base, H_flow = self.train_forward(x, y)
        y_hat_1 = self.fx(x, v_hat_1)

        neg_entropy = -H_base - H_flow

        params_loss = self.param_loss_fn(v, v_hat_1).mean()

        results_tensor = self.test_procedure(
            target=y, estimate=y_hat_1, name="Estimated"
        )
        self.test_procedure(target=y, estimate=x, name="Input")

        v_hat_2, H_base, H_flow = self.train_forward(x, y)
        y_hat_2 = self.fx(x, v_hat_2)

        neg_entropy = neg_entropy - H_base - H_flow
        neg_entropy = neg_entropy / 2

        params_loss = params_loss + self.param_loss_fn(v, v_hat_1).mean()
        params_loss = params_loss / 2

        results_tensor = self.test_procedure(
            target=y_hat_2, estimate=y_hat_1, name="Consistency"
        )

        entropy_bits = (
            -neg_entropy
            * (torch.ones(1, device=neg_entropy.device).exp().log2())
            * self.params_dim
        )

        self.log("Test/Estimated/Params", params_loss)
        self.log("Test/Estimated/entropy_bits", entropy_bits)
        self.log("Test/Estimated/entropy_bits_per_dim", entropy_bits / self.params_dim)

        return results_tensor

    def compute_losses_optim(
        self, x: Tensor, y: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        dim = self.params_dim

        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        z, H_base, mix = self.base_distrib.sample_entropy_mixing(embedding)

        mu = self.base_distrib.get_mu(embedding)
        sigma = self.base_distrib.get_sigma(embedding)

        K = mix.size(1)

        c = c.unsqueeze(1).expand(bs, K, self.context_size).flatten(0, 1)
        z = z.flatten(0, 1)

        H_flow = 0
        for idx, layer in enumerate(self.flow_layers):
            z, ld = layer.forward_and_logdet(z, c)
            H_flow = H_flow + ld.mean(1)
            # self.log(f"check/-H_layer_{idx}", -ld.mean())
        z, ld = self.out.forward_and_logdet(z)
        # self.log(f"check/-H_out", -ld.mean())
        H_flow = H_flow + ld.mean(1)
        zT = z.clone()

        bs, _, N = x.size()
        x = x.expand(bs, K, N).flatten(0, 1).unsqueeze(1)
        y = y.expand(bs, K, N).flatten(0, 1).unsqueeze(1)

        y_hat = self.fx(x, zT)

        audio_loss = self.audio_loss_fn(y_hat, y).view(bs, K)
        audio_loss = (audio_loss * mix).sum(1).mean(0)

        H_flow = H_flow.view(bs, K)
        H_flow = (H_flow * mix).sum(1).mean(0)

        neg_entropy = -H_base.mean() - H_flow
        neg_entropy = neg_entropy / self.params_dim

        v = v.unsqueeze(1).expand(bs, K, v.size(1)).flatten(0, 1)
        params_loss = self.param_loss_fn(v, zT).view(bs, K)
        params_loss = (params_loss * mix).sum(1).mean(0)

        # self.log("check/-H_base", -H_base.mean())
        # self.log("check/-H_flow", -H_flow.mean())
        # self.log("check/mix/max", mix.amax(1).mean())
        # self.log("check/mix/min", mix.amin(1).mean())

        return audio_loss, neg_entropy, params_loss, embedding

    def to(self, device: torch.device):
        self.syn_fx.to(device)
        super().to(device)
        return self

    def cpu(self):
        self.syn_fx.cpu()
        super().cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.syn_fx.cuda(device)
        super().cuda(device)
        return self
