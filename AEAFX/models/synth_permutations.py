# Most of this code was copied from Ben Haye's repo synth-permutations
# https://github.com/ben-hayes/synth-permutations


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
from .main import BEAFX

# from .flow_layers import PlanarLayer, SimplePlanarLayer, SigmoidLayer, SigmoidLayer

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import gamma


def compute_distance_matrix(x0: Tensor, x1: Tensor) -> Tensor:
    x0 = x0.unsqueeze(1)
    x1 = x1.unsqueeze(0)

    M = (x0 - x1).square().sum(2).sqrt()
    return M


def linear_sum_match(x0: Tensor, x1: Tensor):
    cost = compute_distance_matrix(x0, x1).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    x0 = x0[row_ind]
    x1 = x1[col_ind]
    return x0, x1


def nball_volume(d: int, R: float = 1.0):
    num = np.float_power(np.pi, d / 2) * np.power(R, d)
    den = gamma(d / 2 + 1)
    return num / den


def MMD_kernel(x: Tensor, y: Tensor, C: float = 1) -> Tensor:
    return C / (C + (x - y).square().sum(-1))


class EquivProj(nn.Module):
    def __init__(
        self,
        d_token: int,
        num_params: int,
        num_tokens: int,
        params_to_tokens_map: list = [[0, 1, 6], [2, 3, 7], [4, 5, 8]],
    ):
        super().__init__()

        assignment = torch.zeros((num_tokens, num_params))
        for tok_idx in range(len(params_to_tokens_map)):
            for param_idx in params_to_tokens_map[tok_idx]:
                assignment[tok_idx, param_idx] = 1.0 / np.sqrt(num_tokens * num_params)

        self._assignment = nn.Parameter(assignment, requires_grad=False)

        # proj = torch.randn(1, d_token) / np.sqrt(d_token)
        # proj = proj.repeat(num_params, 1)
        # proj = proj + 1e-4 * torch.randn_like(proj)

        proj = torch.randn(num_params, 1) / np.sqrt(d_token)
        proj = proj.repeat(1, d_token)
        proj = proj + 1e-4 * torch.randn_like(proj)

        self._in_projection = nn.Parameter(proj.clone())
        self._out_projection = nn.Parameter(proj.T.clone())

    @property
    def assignment(self):
        return self._assignment

    @property
    def in_projection(self):
        return self._in_projection

    @property
    def out_projection(self):
        return self._out_projection

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("bn,nd->bnd", x, self.in_projection)

        tokens = torch.einsum("bnd,kn->bkd", values, self.assignment)

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)

        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        # we apply L1 penalty to the assignment matrix
        penalty = self.assignment.abs().mean()

        return 0


class LearntProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_token: int,
        num_params: int,
        num_tokens: int,
        initial_ffn: bool = True,
        final_ffn: bool = True,
    ):
        super().__init__()

        assignment = torch.full(
            (num_tokens, num_params), 1.0 / np.sqrt(num_tokens * num_params)
        )
        assignment = assignment + 1e-4 * torch.randn_like(assignment)
        self._assignment = nn.Parameter(assignment)

        proj = torch.randn(1, d_token) / np.sqrt(d_token)
        proj = proj.repeat(num_params, 1)
        proj = proj + 1e-4 * torch.randn_like(proj)

        self._in_projection = nn.Parameter(proj.clone())
        self._out_projection = nn.Parameter(proj.T.clone())

        if initial_ffn:
            self.initial_ffn = nn.Sequential(
                nn.Linear(d_token, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.initial_ffn = None

        if final_ffn:
            self.final_ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_token),
            )
        elif d_token == d_model:
            self.final_ffn = None
        else:
            self.final_ffn = nn.Linear(d_model, d_token)

    @property
    def assignment(self):
        return self._assignment

    @property
    def in_projection(self):
        return self._in_projection

    @property
    def out_projection(self):
        return self._out_projection

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("bn,nd->bnd", x, self.in_projection)

        if self.initial_ffn is not None:
            values = self.initial_ffn(values)

        tokens = torch.einsum("bnd,kn->bkd", values, self.assignment)

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)

        if self.final_ffn is not None:
            deassigned = self.final_ffn(deassigned)

        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        # we apply L1 penalty to the assignment matrix
        penalty = self.assignment.abs().mean()

        return penalty


class DiTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        conditioning_dim: int,
        num_heads: int,
        d_ff: int,
        norm: Literal["layer", "rms"] = "layer",
        first_norm: bool = True,
        adaln_mode: Literal["basic", "zero", "res"] = "basic",
        zero_init: bool = True,
    ):
        super().__init__()
        if first_norm:
            self.norm1 = (
                nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
            )
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # self.attn = MultiheadAttention(d_model, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        cond_out_dim = d_model * 6 if adaln_mode != "res" else d_model * 4
        self.adaln_mode = adaln_mode
        self.cond = nn.Sequential(
            nn.GELU(),
            nn.Linear(conditioning_dim, cond_out_dim),
        )

        self._init_adaln(adaln_mode)
        self._init_ffn(zero_init)
        self._init_attn(zero_init)

    def _init_adaln(self, mode: Literal["basic", "zero"]):
        if mode == "zero":
            nn.init.constant_(self.cond[-1].weight, 0.0)
            nn.init.constant_(self.cond[-1].bias, 0.0)

    def _init_ffn(self, zero_init: bool):
        nn.init.xavier_normal_(self.ff[0].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.zeros_(self.ff[-1].bias)

        if zero_init:
            nn.init.zeros_(self.ff[-1].weight)
        else:
            nn.init.xavier_normal_(self.ff[-1].weight)

    def _init_attn(self, zero_init: bool):
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, z) -> torch.Tensor:
        if self.adaln_mode == "res":
            g1, b1, g2, b2 = self.cond(z)[:, None].chunk(4, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = self.cond(z)[:, None].chunk(6, dim=-1)

        res = x
        x = self.norm1(x)
        x = g1 * x + b1
        x = self.attn(x, x, x)[0]

        if self.adaln_mode == "res":
            x = x + res
        else:
            x = a1 * x + res

        res = x
        x = self.norm2(x)
        x = g2 * x + b2
        x = self.ff(x)

        if self.adaln_mode == "res":
            x = x + res
        else:
            x = a2 * x + res

        return x


class LearntProjVectorField(nn.Module):
    def __init__(
        self,
        num_params,
        token_size: int,
        num_tokens: int,
        model_size: int,
        num_layers: int,
        conditioning_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.projection = LearntProjection(
            d_model=model_size,
            d_token=token_size,
            num_params=num_params,
            num_tokens=num_tokens,
        )

        self.diff_tfm = nn.Sequential()
        for i in range(num_layers):
            self.diff_tfm.append(
                DiTransformerBlock(
                    d_model=model_size,
                    conditioning_dim=conditioning_dim + 1,
                    d_ff=model_size * 2,
                    num_heads=4,
                    adaln_mode="basic",
                )
            )

    def forward(self, vt: Tensor, cond: Tensor, t: Tensor) -> Tensor:

        cond_t = torch.cat((cond, t), dim=1)

        tok_t = self.projection.param_to_token(vt)

        for tfm_layer in self.diff_tfm:
            tok_t = tfm_layer(tok_t, cond_t)

        vt_out = self.projection.token_to_param(tok_t)

        return vt_out


class FFNectorField(nn.Module):
    def __init__(
        self,
        num_params,
        model_size: int,
        num_layers: int,
        conditioning_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            FCBlock(
                input_dim=num_params + conditioning_dim + 1,
                output_dim=model_size,
                bn=False,
                dropout=dropout,
            )
        )

        for i in range(num_layers):
            self.net.append(ResBlock(dim=model_size, bn=False, dropout=dropout))

        self.net.append(
            FCBlock(
                input_dim=model_size,
                output_dim=num_params,
                bn=False,
            )
        )

    def forward(self, vt: Tensor, cond: Tensor, t: Tensor) -> Tensor:

        net_input = torch.cat((vt, cond, t), dim=1)

        vt_out = self.net(net_input)

        return vt_out


class SynthPerm(BEAFX):
    def __init__(
        self,
        fx: DDAFX,
        frontend_args: dict[str, str],
        mlp_depth: int,
        mlp_size: int,
        mlp_type: Literal["mlp", "res"],
        mlp_bn: bool,
        vector_field_args: dict,
        dropout: float = 0.0,
        audio_loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: Callable[[], dict] = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        minibatch_ot: bool = False,
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

        self.dropout = dropout
        self.mlp_bn = mlp_bn
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.mlp_type = mlp_type

        self.minibatch_ot = minibatch_ot

        self.proj_is_p2t = vector_field_args["type"] == "p2t"

        frontend = self.get_frontend(args=frontend_args)

        mlp = self.init_MLP(frontend.out_dim)
        self.encoder = nn.Sequential(frontend, mlp)

        self.vector_field = self.get_vector_field(vector_field_args)

    def init_MLP(self, frontend_dim):
        MLP = nn.Sequential()
        bn = self.mlp_bn

        if frontend_dim != self.mlp_size:
            MLP.append(
                FCBlock(frontend_dim, self.mlp_size, dropout=self.dropout, bn=bn)
            )

        for i in range(self.mlp_depth):
            if self.mlp_type == "mlp":
                MLP.append(
                    FCBlock(self.mlp_size, self.mlp_size, dropout=self.dropout, bn=bn)
                )
            elif self.mlp_type == "res":
                MLP.append(ResBlock(self.mlp_size, dropout=self.dropout, bn=bn))
        return MLP

    def get_vector_field(self, args: dict):
        if args["type"] == "p2t":
            vf = LearntProjVectorField(
                num_params=self.fx.num_parameters,
                num_tokens=args["num_tokens"],
                token_size=args["token_size"],
                model_size=args["model_size"],
                num_layers=args["num_layers"],
                conditioning_dim=self.mlp_size,
            )
        elif args["type"] == "ffn":
            vf = FFNectorField(
                num_params=self.fx.num_parameters,
                model_size=args["model_size"],
                num_layers=args["num_layers"],
                conditioning_dim=self.mlp_size,
            )

        return vf

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        bs = x.size(0)
        v_hat = self.get_FXParams(y=y)

        y_hat = self.fx(x, v_hat)
        return y_hat

    def RK4_step(
        self, vt: Tensor, cond: Tensor, t: Tensor, h: float
    ) -> tuple[Tensor, Tensor]:
        k1 = self.vector_field(vt, cond, t)
        k2 = self.vector_field(vt + h / 2 * k1, cond, t + h / 2)
        k3 = self.vector_field(vt + h / 2 * k2, cond, t + h / 2)
        k4 = self.vector_field(vt + h * k3, cond, t * h)

        v_out = vt + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_out = t + h
        return v_out, t_out

    def get_FXParams(
        self, y: Tensor, num_steps: int = 100, num_samples: int = 1
    ) -> Tensor:
        bs = y.size(0)
        device = y.device
        embedding: Tensor = self.encoder(y)

        embedding = embedding.unsqueeze(1).expand(bs, num_samples, -1).flatten(0, 1)

        vt = torch.rand(bs * num_samples, self.fx.num_parameters, device=device)
        t = torch.zeros(bs * num_samples, 1, device=device)

        h = 1 / num_steps

        for step_idx in range(num_steps):
            vt, t = self.RK4_step(vt, embedding, t, h)

        v1 = torch.clip(vt, min=0, max=1)

        if num_samples != 1:
            v1 = v1.unflatten(0, (bs, num_samples))

        return v1

    def get_KLentropy_MMD_estimates(self, y: Tensor, K: int) -> tuple[Tensor, Tensor]:
        bs = y.size(0)
        d = self.fx.num_parameters

        z = self.get_FXParams(y, num_steps=100, num_samples=K)

        ####################################
        # Computing the entropy using the Kozachenko Leonenko estimator

        z0 = z.unsqueeze(1)
        z1 = z.unsqueeze(2)
        R, _ = (z0 - z1).square().sum(3).sqrt().sort(dim=2)
        R = R[:, :, 1]
        Y = K * torch.pow(R, d)
        H_KL = torch.mean(torch.log(Y), dim=1) + 0.577 + np.log(nball_volume(d))

        ###################################
        # Computing the MMD
        C = 0.5 * d
        z_prior = torch.rand_like(z)
        mask = torch.ones(K, K, device=y.device) - torch.eye(K, K, device=y.device)
        mask = mask.unsqueeze(0) / (K * (K - 1))

        MMD = -2 * MMD_kernel(z.unsqueeze(1), z_prior.unsqueeze(2), C).mean((1, 2))

        temp = MMD_kernel(z.unsqueeze(1), z.unsqueeze(2), C)
        temp = temp * mask
        MMD = MMD + temp.sum((1, 2))

        temp = MMD_kernel(z_prior.unsqueeze(1), z_prior.unsqueeze(2), C)
        temp = temp * mask
        MMD = MMD + temp.sum((1, 2))

        return H_KL, MMD

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v1 = batch
        bs = x.size(0)

        cond: Tensor = self.encoder(y)

        v0 = torch.rand_like(v1)

        if self.minibatch_ot:
            v0, v1 = linear_sum_match(v0, v1)

        t = torch.rand(bs, 1, device=v0.device)
        vt = v1 * t + v0 * (1 - t)

        target: Tensor = v1 - v0
        est: Tensor = self.vector_field(vt, cond, t)

        diff_loss = (target - est).square().mean(1).mean(0) * 10

        total_loss = diff_loss
        if self.proj_is_p2t:
            self.vector_field: LearntProjVectorField
            penalty = self.vector_field.projection.penalty()

            self.log("loss_penalty/train", penalty)
            total_loss = total_loss + penalty * 0.05

        self.log("loss_diff/train", diff_loss)
        self.log("loss_total/train", total_loss)
        return total_loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx):
        x, y, v = batch

        z = self.get_FXParams(y, num_steps=16)

        yhat = self.fx(x, z)

        audio_loss = self.audio_loss_fn(y, yhat).mean()

        if v.size(1) == z.size(1):
            params_loss = (v - z).square().mean(1).mean(0)
        else:
            params_loss = 0.0

        self.log("loss_audio/valid", audio_loss)
        self.log("loss_params/valid", params_loss)
        self.log("loss_total/valid", audio_loss)
        return audio_loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch

        z = self.get_FXParams(y, num_steps=100)

        yhat = self.fx(x, z)

        audio_loss = self.audio_loss_fn(y, yhat).mean()

        if v.size(1) == z.size(1):
            params_loss = (v - z).square().mean(1).mean(0)
        else:
            params_loss = 0.0

        self.log("loss_audio/test", audio_loss)
        self.log("loss_params/test", params_loss)
        self.log("loss_total/test", audio_loss)


class SelfGen_Synthperm(SynthPerm):
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
