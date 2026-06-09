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


class BEAFX(pl.LightningModule):
    def __init__(
        self,
        fx: DDAFX,
        loss_fn: Callable[[Tensor], Tensor],
        metrics_dict: dict = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        audio_or_params: Literal["audio", "params"] = "params",
    ):
        super().__init__()
        self.lr_sched_patience = lr_sched_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fx = fx
        self.loss_fn = loss_fn
        self.audio_or_params = audio_or_params

        if metrics_dict is None:
            metrics_dict = {}
        self.metrics_dict = metrics_dict

    def param_loss_fn(self, v_hat: Tensor, v: Tensor):
        if v.size() == v_hat.size():
            return torch.mean(torch.square(v_hat - v), dim=1)
        else:
            return torch.zeros(v.size(0), device=v.device)

    def test_procedure(
        self,
        target: Tensor,
        estimate: Tensor,
        name: Literal["Estimated", "Input", "Consistency"],
    ):
        results_tensor = torch.zeros(len(self.metrics_dict))
        for i, (metric_name, metric_fn) in enumerate(self.metrics_dict.items()):
            estimate_metric: Tensor = metric_fn(estimate, target).mean()
            self.log(f"Test/{name}/{metric_name}", estimate_metric)
            results_tensor[i] = estimate_metric

        return results_tensor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        out_dict = {
            "optimizer": optimizer,
        }

        if self.lr_sched_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                patience=self.lr_sched_patience,
                threshold=0.02,
            )
            out_dict["lr_scheduler"] = scheduler
            out_dict["monitor"] = f"loss_audio_mean/valid"

        return out_dict

    def to(self, device: torch.device):
        self.fx.to(device)
        super().to(device)
        return self

    def cpu(self):
        self.fx.cpu()
        super().cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.fx.cuda(device)
        super().cuda(device)
        return self

    def get_frontend(self, args: dict[str, str]):

        if args["type"].lower() == "tfe":
            fe = frontend.TimeFrequencyCQT_Encoder(
                samplerate=args["samplerate"],
                n_bins=args["n_bins"],
                out_dim=args["out_dim"],
            )
        if args["type"].lower() == "simple":
            fe = frontend.SimpleFrontend(
                out_channels=args["out_channels"],
                kernel_size=args["kernel_size"],
                stride=args["stride"],
            )
        if args["type"].lower() == "melnext":
            fe = frontend.MelNeXt(
                kernel_size=args["kernel_size"],
                channels_list=args["channels_list"],
                out_dim=args["out_dim"],
                n_mels=args["n_mels"],
                convnext_bottleneck_factor=args["convnext_bottleneck_factor"],
                samplerate=args["samplerate"],
            )
        if args["type"].lower() == "melnext_att":
            fe = frontend.MelNeXt_Attention(
                kernel_size=args["kernel_size"],
                channels_list=args["channels_list"],
                out_dim=args["out_dim"],
                n_mels=args["n_mels"],
                convnext_bottleneck_factor=args["convnext_bottleneck_factor"],
                samplerate=args["samplerate"],
            )    
        return fe


class FX_AE_aMCL(BEAFX):
    def __init__(
        self,
        frontend_args: dict[str, str],
        fx: DDAFX,
        mlp_depth: tuple[int] = 2,
        mlp_size: int = 64,
        mlp_type: Literal["mlp", "res"] = "mlp",
        mlp_bn: bool = True,
        num_hyp: int = 4,
        dropout: float = 0.0,
        loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: dict[str, Callable[[Tensor], Tensor]] = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        annealing_length: float = 300,
        start_T: float = 10,
        temp_decay: float = 0.998,
        min_T: float = 1e-2,
        audio_or_params: Literal["audio", "params"] = "params",
    ):
        super().__init__(
            fx=fx,
            metrics_dict=metrics_dict,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
            audio_or_params=audio_or_params,
        )
        self.save_hyperparameters(ignore=["metrics_dict", "loss_fn"])

        self.mlp_depth = mlp_depth
        self.mlp_size = mlp_size
        self.mlp_type = mlp_type
        self.mlp_bn = mlp_bn

        self.dropout = dropout
        self.num_hyp = num_hyp
        self.start_T = start_T
        self.temp_decay = temp_decay
        self.annealing_length = annealing_length

        self.temp = start_T
        self.min_T = min_T

        self.frontend = self.get_frontend(args=frontend_args)
        self.base_mlp = self.init_base()

        self.output_layers = nn.Sequential(
            nn.Linear(self.mlp_size, self.fx.num_parameters * self.num_hyp),
            nn.Unflatten(1, (self.num_hyp, self.fx.num_parameters)),
            nn.Sigmoid(),
        )

        self.softmin = nn.Softmin(dim=1)

    def init_base(self):
        out = nn.Sequential()
        if self.frontend.out_dim != self.mlp_size:
            out.append(
                FCBlock(
                    self.frontend.out_dim,
                    self.mlp_size,
                    dropout=self.dropout,
                    bn=self.mlp_bn,
                ),
            )

        for idx in range(self.mlp_depth):
            if self.mlp_type == "mlp":
                out.append(
                    FCBlock(
                        self.mlp_size,
                        self.mlp_size,
                        dropout=self.dropout,
                        bn=self.mlp_bn,
                    )
                )
            elif self.mlp_type == "res":
                out.append(
                    ResBlock(dim=self.mlp_size, dropout=self.dropout, bn=self.mlp_bn)
                )

        return out

    def get_scores(self, embedding: Tensor):
        bs = embedding.size(0)
        device = embedding.device

        scores = torch.ones(bs, self.num_hyp, device=device) / self.num_hyp
        return scores

    def get_hyps_and_scores(self, y) -> tuple[Tensor, Tensor]:
        e = self.base_mlp(self.frontend(y))

        hyps = self.output_layers(e)

        scores = self.get_scores(e)

        return hyps, scores

    def get_q(self, loss_params: Tensor) -> Tensor:

        temp = self.temp
        # eps=5e-4
        eps = self.min_T

        # if temp < eps:
        #     min_idx = torch.argmin(loss_params, dim=1, keepdim=True)
        #     q = torch.zeros_like(loss_params)
        #     ones = torch.ones_like(loss_params)
        #     q = q.scatter(index=min_idx, dim=1, src=ones)
        # else:
        q = self.softmin(loss_params / temp)
        return q

    def get_q_and_scoring_target(self, l: Tensor) -> tuple[Tensor, Tensor]:
        temp = self.temp
        # eps=5e-4
        eps = self.min_T

        q:Tensor = self.softmin(l / temp)

        scoring_target = torch.argmin(l, dim=1, keepdim=False)
        # scoring_target:Tensor = torch.nn.functional.one_hot(scoring_target, num_classes=self.num_hyp)
        # scoring_target = scoring_target.to(device=q.device, dtype=q.dtype)


        return q, scoring_target

    def compute_params_loss(self, z: Tensor, v: Tensor):
        loss_params = torch.mean(torch.square(z - v), dim=(2))
        q = self.get_q(loss_params)
        loss_params = (loss_params * q).sum(1).mean()
        return loss_params

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        ramp_ratio = min(self.trainer.current_epoch / self.annealing_length, 1)
        self.temp = np.exp(
            np.log(self.start_T) * (1 - ramp_ratio) + np.log(self.min_T) * ramp_ratio
        )

        mode = "train"
        x, y, v = batch
        bs, _, N = x.size()
        z, scores = self.get_hyps_and_scores(y)

        if self.audio_or_params == "audio":
            z = z.flatten(0, 1)
            x = x.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            y_hat = self.fx(x, z)
            y = y.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            loss_audio: Tensor = self.loss_fn(y_hat, y)

            loss_audio = loss_audio.unflatten(0, (bs, self.num_hyp))
            q, scoring_target = self.get_q_and_scoring_target(loss_audio)
            q = q.detach()

            loss_audio_aMCL = (loss_audio * q).sum(1).mean()
            loss_audio_oracle = loss_audio.amin(1).mean()
            loss_audio_mean = (loss_audio * scores).sum(1).mean()
            loss_scoring = torch.nn.functional.nll_loss(
                scores, scoring_target, reduction="mean"
            )

            out = loss_audio_aMCL + loss_scoring
            self.log(f"loss_audio_aMCL/{mode}", loss_audio_aMCL)
            self.log(f"loss_audio_oracle/{mode}", loss_audio_oracle)
            self.log(f"loss_audio_mean/{mode}", loss_audio_mean)
            self.log(f"scoring_loss/{mode}", loss_scoring)

        elif self.audio_or_params == "params":
            v = v.unsqueeze(1)
            loss_params = torch.mean(torch.square(z - v), dim=(2))
            q = self.get_q(loss_params)
            loss_params = (loss_params * q).sum(1).mean()
            v = v.squeeze(1)
            out = loss_params * 10
            self.log(f"loss_params/{mode}", loss_params)

        else:
            raise (ValueError(f"audio_or_params error : {self.audio_or_params}"))

        self.log(f"loss_total/{mode}", out)
        self.log(f"q_min/{mode}", torch.amin(q, dim=1).mean())
        self.log(f"q_max/{mode}", torch.amax(q, dim=1).mean())

        self.log(f"score_max/{mode}", torch.amax(scores, dim=1).mean())
        self.log(f"score_min/{mode}", torch.amin(scores, dim=1).mean())
        self.log(f"temp/{mode}", self.temp)

        return out

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):

        mode = "valid"
        x, y, v = batch
        bs, _, N = x.size()
        z, scores = self.get_hyps_and_scores(y)

        if self.audio_or_params == "audio":
            z = z.flatten(0, 1)
            x = x.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            y_hat = self.fx(x, z)
            y = y.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            loss_audio: Tensor = self.loss_fn(y_hat, y)

            loss_audio = loss_audio.unflatten(0, (bs, self.num_hyp))
            q, scoring_target = self.get_q_and_scoring_target(loss_audio)
            q = q.detach()

            loss_audio_aMCL = (loss_audio * q).sum(1).mean()
            loss_audio_oracle = loss_audio.amin(1).mean()
            loss_audio_mean = (loss_audio * scores).sum(1).mean()
            loss_scoring = torch.nn.functional.nll_loss(
                scores, scoring_target, reduction="mean"
            )

            out = loss_audio_aMCL + loss_scoring
            self.log(f"loss_audio_aMCL/{mode}", loss_audio_aMCL)
            self.log(f"loss_audio_oracle/{mode}", loss_audio_oracle)
            self.log(f"loss_audio_mean/{mode}", loss_audio_mean)
            self.log(f"scoring_loss/{mode}", loss_scoring)

        elif self.audio_or_params == "params":
            v = v.unsqueeze(1)
            loss_params = torch.mean(torch.square(z - v), dim=(2))
            q = self.get_q(loss_params)
            loss_params = (loss_params * q).sum(1).mean()
            v = v.squeeze(1)
            out = loss_params * 10
            self.log(f"loss_params/{mode}", loss_params)

        else:
            raise (ValueError(f"audio_or_params error : {self.audio_or_params}"))

        self.log(f"loss_total/{mode}", out)
        self.log(f"q_min/{mode}", torch.amin(q, dim=1).mean())
        self.log(f"q_max/{mode}", torch.amax(q, dim=1).mean())
        self.log(f"score_max/{mode}", torch.amax(scores, dim=1).mean())
        self.log(f"score_min/{mode}", torch.amin(scores, dim=1).mean())
        self.log(f"temp/{mode}", self.temp)

        return out

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        mode = "test"
        x, y, v = batch
        bs, _, N = x.size()
        z, scores = self.get_hyps_and_scores(y)

        if self.audio_or_params == "audio":
            z = z.flatten(0, 1)
            x = x.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            y_hat = self.fx(x, z)
            y = y.unsqueeze(1).expand(bs, self.num_hyp, 1, N).flatten(0, 1)
            loss_audio: Tensor = self.loss_fn(y_hat, y)

            loss_audio = loss_audio.unflatten(0, (bs, self.num_hyp))
            q, scoring_target = self.get_q_and_scoring_target(loss_audio)
            q = q.detach()

            loss_audio_aMCL = (loss_audio * q).sum(1).mean()
            loss_audio_oracle = loss_audio.amin(1).mean()
            loss_audio_mean = (loss_audio * scores).sum(1).mean()
            loss_scoring = torch.nn.functional.nll_loss(
                scores, scoring_target, reduction="mean"
            )

            out = loss_audio_aMCL + loss_scoring
            self.log(f"loss_audio_aMCL/{mode}", loss_audio_aMCL)
            self.log(f"loss_audio_oracle/{mode}", loss_audio_oracle)
            self.log(f"loss_audio_mean/{mode}", loss_audio_mean)
            self.log(f"scoring_loss/{mode}", loss_scoring)

        elif self.audio_or_params == "params":
            v = v.unsqueeze(1)
            loss_params = torch.mean(torch.square(z - v), dim=(2))
            q = self.get_q(loss_params)
            loss_params = (loss_params * q).sum(1).mean()
            v = v.squeeze(1)
            out = loss_params * 10
            self.log(f"loss_params/{mode}", loss_params)

        else:
            raise (ValueError(f"audio_or_params error : {self.audio_or_params}"))

        self.log(f"loss_total/{mode}", out)
        self.log(f"q_min/{mode}", torch.amin(q, dim=1).mean())
        self.log(f"q_max/{mode}", torch.amax(q, dim=1).mean())
        self.log(f"score_max/{mode}", torch.amax(scores, dim=1).mean())
        self.log(f"score_min/{mode}", torch.amin(scores, dim=1).mean())
        self.log(f"temp/{mode}", self.temp)

        


class FX_AE_arMCL(FX_AE_aMCL):
    def __init__(
        self,
        frontend_args: dict[str, str],
        fx: DDAFX,
        mlp_depth: tuple[int] = 2,
        mlp_size: int = 64,
        mlp_type: Literal["mlp", "res"] = "mlp",
        mlp_bn: bool = True,
        num_hyp: int = 4,
        dropout: float = 0.0,
        loss_fn: Callable[[Tensor], Tensor] = None,
        metrics_dict: dict[str, Callable[[Tensor], Tensor]] = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
        annealing_length: float = 300,
        start_T: float = 10,
        temp_decay: float = 0.998,
        min_T: float = 1e-2,
        audio_or_params: Literal["audio", "params"] = "params",
    ):
        super().__init__(
            frontend_args=frontend_args,
            fx=fx,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            mlp_bn=mlp_bn,
            num_hyp=num_hyp,
            dropout=dropout,
            loss_fn=loss_fn,
            metrics_dict=metrics_dict,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
            annealing_length=annealing_length,
            start_T=start_T,
            temp_decay=temp_decay,
            min_T=min_T,
            audio_or_params=audio_or_params,
        )

        self.scoring_heads = nn.Sequential(
            nn.Linear(self.mlp_size,  self.num_hyp),
            nn.Softmax(dim=1),
        )

    def get_scores(self, e):
        return self.scoring_heads(e)


# class SelfGen_AE(FX_AE):
#     def training_step(self, batch, batch_idx):
#         x, v = batch
#         self.fx.synthesis()
#         y = self.fx(x, v)
#         self.fx.analysis()
#         return super().training_step((x, y, v), batch_idx)

#     def validation_step(self, batch, batch_idx):
#         x, v = batch
#         self.fx.synthesis()
#         y = self.fx(x, v)
#         self.fx.analysis()
#         return super().validation_step((x, y, v), batch_idx)

#     def test_step(self, batch, batch_idx):
#         x, v = batch
#         self.fx.synthesis()
#         y = self.fx(x, v)
#         self.fx.analysis()
#         return super().test_step((x, y, v), batch_idx)
