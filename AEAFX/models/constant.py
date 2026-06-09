import torch.utils
import torch.utils.data
import torch
import torch.nn as nn
import lightning.pytorch as pl
from ..ddafx import DDAFX
from ..utils import safe_log
from typing import Optional, Union, Literal, LiteralString, Callable
from ..loss import abs_params_loss
from .distributions import ConstantDistribution, ConstantGaussian, ConstantGMMFull
from torch import Tensor


# from .flow_layers import PlanarLayer, SimplePlanarLayer, SigmoidLayer, SigmoidLayer

from . import flows
import numpy as np
from .main import BEAFX


class ConstantParams(BEAFX):
    def __init__(
        self,
        fx: DDAFX,
        loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: dict = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        audio_loss_weight: float = 1.0,
        params_loss_weight: float = 0.0,
    ):
        super().__init__(
            fx=fx,
            metrics_dict=metrics_dict,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters(ignore=["metrics_dict", "loss_fn"])

        self.audio_loss_weight = audio_loss_weight
        self.params_loss_weight = params_loss_weight

        self.unnorm_params = nn.Parameter(torch.zeros(1, self.fx.num_parameters))
        self.sig = nn.Sigmoid()

    def get_FXParams(self) -> Tensor:
        return self.sig(self.unnorm_params)

    def forward(self, x: Tensor, y: Tensor):
        zhat = self.get_FXParams()
        yh = self.fx(x, zhat)
        return yh

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        bs = x.size(0)

        z = self.get_FXParams().expand(bs, self.fx.num_parameters)
        y_hat = self.fx(x, z)

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()

        total_loss = self.audio_loss_weight * loss_audio
        total_loss = total_loss + self.params_loss_weight * loss_params

        self.log("loss_audio/train", loss_audio)
        self.log("loss_total/train", total_loss)
        self.log("loss_params/train", loss_params)
        return total_loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        bs = x.size(0)

        z = self.get_FXParams().expand(bs, self.fx.num_parameters)
        y_hat = self.fx(x, z)

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()

        total_loss = self.audio_loss_weight * loss_audio
        total_loss = total_loss + self.params_loss_weight * loss_params

        self.log("loss_audio/valid", loss_audio)
        self.log("loss_total/valid", total_loss)
        self.log("loss_params/valid", loss_params)
        return total_loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        bs = x.size(0)

        z = self.get_FXParams().expand(bs, self.fx.num_parameters)
        y_hat = self.fx(x, z)

        loss_params = self.param_loss_fn(v, z).mean()

        self.test_procedure(target=y, estimate=y_hat, name=["Estimated", "Best"])
        self.test_procedure(target=y, estimate=x, name="Input")
        self.log("Test/Estimated/Params", loss_params)


class ConstantInference(BEAFX):
    def __init__(
        self,
        fx: DDAFX,
        start_beta: float = 1,
        end_beta: float = 1,
        warmup_length: int = 15,
        audio_loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: Callable[[], dict] = None,
        flow_length: int = 5,
        flow_layers_type: str = "static",
        flow_nl: str = "res",
        flow_nl_knots: int = 4,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        num_mixtures: int = 10,
        base_entropy: Literal["direct", "MC"] = "direct",
        num_tries_best: int = 2,
        audio_loss_weight: float = 1.0,
        params_loss_weight: float = 0.0,
    ):
        super().__init__(
            fx,
            audio_loss_fn,
            metrics_dict,
            learning_rate,
            lr_sched_patience,
            weight_decay,
        )
        self.save_hyperparameters(ignore=["metrics_dict", "audio_loss_fn"])

        self.audio_loss_fn = audio_loss_fn

        params_dim = fx.num_parameters
        self.params_dim = params_dim

        self.start_beta = start_beta
        self.end_beta = end_beta
        self.beta = start_beta
        self.warmup_length = warmup_length

        self.num_mixtures = num_mixtures
        self.num_tries_best = num_tries_best

        self.compute_base_entropy = base_entropy

        self.init_flow(
            flow_length=flow_length,
            flow_nl=flow_nl,
            flow_nl_knots=flow_nl_knots,
            flow_layers_type=flow_layers_type
        )

        self.out = flows.SigmoidLayer()

        if self.num_mixtures==1:
            self.base_distrib: ConstantDistribution = ConstantGaussian(
                params_dim, base_entropy=base_entropy
            )
            self.num_mixtures = 1
        else:
            self.base_distrib = ConstantGMMFull(
                params_dim,
                num_mixtures=num_mixtures,
                base_entropy=base_entropy,
            )
    def init_flow(
        self,
        flow_length: int = 5,
        flow_layers_type:str="static",
        flow_nl: str = "dsf",
        flow_nl_knots: int = 4,
    ):
        params_dim = self.params_dim
        nl_bound = 8
        self.flow_layers: nn.Sequential[flows.utils.Flow] = nn.Sequential()
        # self.flow_layers: nn.Sequential[flows.utils.Flow] = nn.Sequential(flows.linear.Rectangular(in_dim=1, out_dim=params_dim))
        for _ in range(flow_length):
            self.flow_layers.append(flows.linear.RandomPermutation(dim=params_dim))

            if flow_layers_type=="static":
                self.flow_layers.append(flows.linear.StaticLower(dim=params_dim))
                self.flow_layers.append(flows.linear.StaticUpper(dim=params_dim, offset=1))
            # self.flow_layers.append(
            #     flows.DSF_Static(
            #         dim=params_dim,
            #         hidden_dim=flow_nl_knots,
            #         gru_num_layers=2,
            #     )
            # )
            self.flow_layers.append(
                flows.planar.Static(dim=params_dim)
            )

    def train_forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        d = self.params_dim

        z, H_base = self.base_distrib.sample_and_entropy(bs)
        z0 = z.clone()
        H_flow = 0
        for layer in self.flow_layers:
            z, ld = layer.forward_and_logdet(z, None)
            H_flow = H_flow + ld.mean(1)
        z, ld = self.out.forward_and_logdet(z)
        H_flow = H_flow + ld.mean(1)
        zT = z.clone()
        zT = zT.expand(bs, d)
        # latent_loss = -H0 - log_det

        v_hat = zT

        return v_hat, H_base.mean() / self.params_dim, H_flow.mean() / self.params_dim

    def forward(self, x, y) -> Tensor:
        # self.fx.to(self.device)

        bs = x.size(0)
        v_hat = self.get_FXParams(bs).expand(bs, self.params_dim)

        y_hat = self.fx(x, v_hat)
        return y_hat

    def get_FXParams(self, bs:int=1) -> Tensor:


        z, _ = self.base_distrib.sample_and_entropy(bs)

        for layer in self.flow_layers:
            z = layer(z, None)
        z = self.out(z)

        return z

    def compute_losses(
        self, x: Tensor, y: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        d = self.params_dim

        z_list, H_base, mix = self.base_distrib.sample_entropy_mixing(bs)

        H_flow_tot = 0
        audio_loss_tot = 0
        params_loss_tot = 0

        num_mixtures = mix.size(1)

        for mix_idx in range(num_mixtures):
            z = z_list[:, mix_idx, :]
            H_flow = 0
            for layer in self.flow_layers:
                z, ld = layer.forward_and_logdet(z, None)
                H_flow = H_flow + ld.mean(1)
            z, ld = self.out.forward_and_logdet(z)
            H_flow = H_flow + ld.mean(1)
            zT = z.clone()

            # latent_loss = -H0 - log_det

            y_hat = self.fx(x, zT)

            audio_loss_tot = audio_loss_tot + (
                self.audio_loss_fn(y_hat, y) * mix[:, mix_idx]
            )

            H_flow_tot = H_flow_tot + (H_flow * mix[:, mix_idx])
            params_loss_tot = params_loss_tot + (
                self.param_loss_fn(v, zT) * mix[:, mix_idx]
            )

        neg_entropy = -H_base.mean() - H_flow_tot.mean()
        neg_entropy = neg_entropy / self.params_dim

        # self.log("check/-H_base", -H_base.mean())
        # self.log("check/-H_flow", -H_flow_tot.mean())
        # self.log("check/mix/max", mix.amax(1).mean())
        # self.log("check/mix/min", mix.amin(1).mean())

        return audio_loss_tot.mean(), neg_entropy.mean(), params_loss_tot.mean()

    def compute_losses_optim(
        self, x: Tensor, y: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        dim = self.params_dim

        z, H_base, mix = self.base_distrib.sample_entropy_mixing(bs)

        K = mix.size(1)

        z = z.expand(bs, self.num_mixtures, self.params_dim).flatten(0, 1)

        H_flow = 0
        for idx, layer in enumerate(self.flow_layers):
            z, ld = layer.forward_and_logdet(z, None)
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

        return audio_loss, neg_entropy, params_loss

    def training_step(self, batch, batch_idx):
        ramp_ratio = min(self.trainer.current_epoch / self.warmup_length, 1)
        self.beta = np.exp(
            np.log(self.start_beta) * (1 - ramp_ratio)
            + np.log(self.end_beta) * ramp_ratio
        )
        # self.beta = self.start_beta * (1 - ramp_ratio) + self.end_beta * ramp_ratio

        x, y, v = batch
        audio_loss, neg_entropy, params_loss = self.compute_losses_optim(x, y, v)

        total_loss = audio_loss + self.beta * neg_entropy

        entropy_bits = (
            -neg_entropy
            * (torch.ones(1, device=neg_entropy.device).exp().log2())
            * self.params_dim
        )

        self.log("loss_audio/train", audio_loss)
        self.log("loss_params/train", params_loss)
        self.log("loss_total/train", total_loss)
        self.log("entropy_bits/train", entropy_bits)
        self.log("entropy_bits_dim/train", entropy_bits / self.params_dim)
        self.log("beta", self.beta)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, v = batch
        audio_loss, neg_entropy, params_loss = self.compute_losses_optim(x, y, v)
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
        self.log("entropy_bits_dim/valid", entropy_bits / self.params_dim)
        self.log("beta", self.beta)
        return total_loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        bs=x.size(0)
        
        self.test_procedure(target=y, estimate=x, name="Input")
        
        # Estimated loss computation

        z_1, H_base, H_flow = self.train_forward(x, y)
        y_hat_1 = self.fx(x, z_1)

        neg_entropy = -H_base - H_flow

        params_loss = self.param_loss_fn(v, z_1).mean()

        # results_tensor = self.test_procedure(
        #     target=y, estimate=y_hat_1, name="Estimated"
        # )

        z_2, H_base, H_flow = self.train_forward(x, y)
        y_hat_2 = self.fx(x, z_2)

        neg_entropy = neg_entropy - H_base - H_flow
        neg_entropy = neg_entropy / 2

        params_loss = params_loss + self.param_loss_fn(v, z_2).mean()
        params_loss = params_loss / 2

        # Consistency loss computation
        results_tensor = self.test_procedure(
            target=y_hat_2, estimate=y_hat_1, name="Consistency"
        )

        z_1 = torch.rand(bs, self.fx.num_parameters, device=x.device)
        y_hat_1 = self.fx(x, z_1)
        z_2 = torch.rand(bs, self.fx.num_parameters, device=x.device)
        y_hat_2 = self.fx(x, z_2)

        # Consistency loss computation from random params
        results_tensor = self.test_procedure(
            target=y_hat_2, estimate=y_hat_1, name="Consistency - Rand"
        )

        entropy_bits = (
            -neg_entropy
            * (torch.ones(1, device=neg_entropy.device).exp().log2())
            * self.params_dim
        )

        self.log("Test/Estimated/Params", params_loss)
        self.log("Test/Estimated/entropy_bits", entropy_bits)
        self.log("Test/Estimated/entropy_bits_dim", entropy_bits / self.params_dim)

        # Best loss computation
        bs = x.size(0)

        num_tries = self.num_tries_best


        z, _ = self.base_distrib.sample_and_entropy(bs*num_tries)
        for layer in self.flow_layers:
            z = layer(z, None)
        z = self.out(z)

        x_exp = x.unsqueeze(1).expand(bs, num_tries, -1, -1).flatten(0, 1)
        y_hat = self.fx(x_exp, z)

        y_exp = y.unsqueeze(1).expand(bs, num_tries, -1, -1).flatten(0, 1)
        audio_loss: Tensor = self.audio_loss_fn(y_hat, y_exp)
        audio_loss = audio_loss.reshape(bs, num_tries)

        results_tensor = self.test_procedure(
            target=y_exp, estimate=y_hat, name="Estimated"
        )

        best_idx = torch.argmin(audio_loss, 1, keepdim=False)
        best_idx = best_idx.view(bs, 1, 1, 1).expand(bs, 1, 1, x.size(2))

        y_hat = y_hat.reshape(bs, num_tries, 1, y_hat.size(2))

        y_best = y_hat.gather(dim=1, index=best_idx)
        y_best = y_best.view(bs, 1, x.size(2))

        self.test_procedure(target=y, estimate=y_best, name="Best")

        # Best params
        v_size = v.size(1)
        v = v.unsqueeze(1).expand(bs, num_tries, v_size).flatten(0, 1)
        params_loss = self.param_loss_fn(v, z)

        params_loss = params_loss.view(bs, num_tries)
        params_loss = params_loss.amin(1).mean()

        self.log("Test/Best/Params", params_loss)

        # Uniform best

        z = torch.rand((bs * num_tries, self.fx.num_parameters), device=x.device)

        x_exp = x.unsqueeze(1).expand(bs, num_tries, -1, -1).flatten(0, 1)
        y_hat = self.fx(x_exp, z)

        y_exp = y.unsqueeze(1).expand(bs, num_tries, -1, -1).flatten(0, 1)
        audio_loss: Tensor = self.audio_loss_fn(y_hat, y_exp)
        audio_loss = audio_loss.reshape(bs, num_tries)

        best_idx = torch.argmin(audio_loss, 1, keepdim=False)
        best_idx = best_idx.view(bs, 1, 1, 1).expand(bs, 1, 1, x.size(2))

        y_hat = y_hat.reshape(bs, num_tries, 1, y_hat.size(2))

        y_best = y_hat.gather(dim=1, index=best_idx)
        y_best = y_best.view(bs, 1, x.size(2))

        self.test_procedure(target=y, estimate=y_best, name="Uniform_best")

        return results_tensor

    def eval(self):
        super().eval()
        if isinstance(self.base_distrib, ConstantGMMFull):
            self.base_distrib.base_entropy = "MC"
        return self

    def train(self, mode=True):
        super().train(mode=mode)
        if isinstance(self.base_distrib, ConstantGMMFull):
            if mode:
                self.base_distrib.base_entropy = self.compute_base_entropy
            else:
                self.base_distrib.base_entropy = "MC"
        return self


class ConstantSelfGenParams(ConstantParams):
    def training_step(self, batch, batch_idx):
        x, v = batch
        self.fx.synthesis()
        y = self.fx(x, v)
        self.fx.analysis()
        return super().training_step((x, y, v), batch_idx)

    def validation_step(self, batch, batch_idx):
        x, v = batch
        self.fx.synthesis()
        y = self.fx(x, v)
        self.fx.analysis()
        return super().validation_step((x, y, v), batch_idx)

    def test_step(self, batch, batch_idx):
        x, v = batch
        self.fx.synthesis()
        y = self.fx(x, v)
        self.fx.analysis()
        return super().test_step((x, y, v), batch_idx)