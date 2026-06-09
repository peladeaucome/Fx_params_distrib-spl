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
    Gaussian_LogSigma,
)
from torch import Tensor
from .layers import Swish, ResBlock, FCBlock

from scipy.special import gamma

# from .flow_layers import PlanarLayer, SimplePlanarLayer, SigmoidLayer, SigmoidLayer

from . import flows
import numpy as np


def nball_volume(d: int, R: float = 1.0):
    num = np.float_power(np.pi, d / 2) * np.power(R, d)
    den = gamma(d / 2 + 1)
    return num / den


def MMD_kernel(x: Tensor, y: Tensor, C: float = 1) -> Tensor:
    return C / (C + (x - y).square().sum(-1))


class BEAFX(pl.LightningModule):
    def __init__(
        self,
        fx: DDAFX,
        loss_fn: Callable[[Tensor], Tensor],
        metrics_dict: dict = None,
        learning_rate: float = 1e-3,
        lr_sched_patience: int = None,
        weight_decay: float = 3e-5,
    ):
        super().__init__()
        self.lr_sched_patience = lr_sched_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fx = fx
        self.loss_fn = loss_fn

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
        name: Union[list[str], Literal["Estimated", "Input", "Consistency", "Best"]],
    ):

        results_tensor = torch.zeros(len(self.metrics_dict) + 1)
        for i, (metric_name, metric_fn) in enumerate(self.metrics_dict.items()):
            if isinstance(metric_fn, nn.Module):
                metric_fn = metric_fn.to(target.device)

            estimate_metric: Tensor = metric_fn(estimate, target).mean()

            if isinstance(name, list):
                for n in name:
                    self.log(f"Test/{n}/{metric_name}", estimate_metric.mean())
            else:
                self.log(f"Test/{name}/{metric_name}", estimate_metric.mean())
            results_tensor[i] = estimate_metric

        if name != "Input":
            loss_fn = self.loss_fn.to(target.device)
            estimate_metric: Tensor = loss_fn(estimate, target).mean()
            self.log(f"Test/{name}/Loss", estimate_metric)
        results_tensor[-1] = estimate_metric
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
            out_dict["monitor"] = "loss_total/valid"

        return out_dict

    def to(self, device: torch.device):
        self.fx.to(device)
        super().to(device)
        return self

    def cpu(self):
        self.fx.cpu()
        super().cpu()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.fx.train(mode)
        return self

    def eval(self):
        return self.train(False)

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
        if args["type"].lower() == "mfccnext":
            fe = frontend.MFCCNeXt(
                kernel_size=args["kernel_size"],
                channels_list=args["channels_list"],
                out_dim=args["out_dim"],
                n_mels=args["n_mels"],
                convnext_bottleneck_factor=args["convnext_bottleneck_factor"],
                samplerate=args["samplerate"],
            )
        if args["type"].lower() == "meanstft":
            fe = frontend.MeanSTFT(n_fft=args["n_fft"], hop_length=args["hop_length"])
        if args["type"].lower() == "meancqt":
            fe = frontend.MeanCQT(
                fmin=args["fmin"],
                fmax=args["fmax"],
                hop_length=args["hop_length"],
                nbins=args["nbins"],
                samplerate=args["samplerate"],
            )
        return fe


class FX_AE(BEAFX):
    def __init__(
        self,
        frontend_args: dict[str, str],
        fx: DDAFX,
        mlp_depth: tuple[int] = 2,
        mlp_size: int = 64,
        mlp_type: Literal["mlp", "res"] = "mlp",
        mlp_bn: bool = True,
        dropout: float = 0.0,
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
        self.save_hyperparameters(ignore=["metrics_dict", "loss_fn", "fx"])

        self.mlp_depth = mlp_depth
        self.mlp_size = mlp_size
        self.mlp_type = mlp_type
        self.mlp_bn = mlp_bn
        self.dropout = dropout

        frontend = self.get_frontend(args=frontend_args)

        self.audio_loss_weight = audio_loss_weight
        self.params_loss_weight = params_loss_weight

        mlp = self.init_mlp(frontend.out_dim)

        self.encoder = nn.Sequential(frontend, mlp)

        self.out_layer = nn.Linear(self.mlp_size, self.fx.num_parameters)
        # self.mlp.append(nn.BatchNorm1d(num_features=self.fx.num_parameters))
        self.sig = nn.Sigmoid()

    def init_mlp(self, in_dim):
        bn = self.mlp_bn
        mlp = nn.Sequential()
        if in_dim != self.mlp_size:
            mlp.append(
                FCBlock(in_dim, self.mlp_size, dropout=self.dropout, bn=bn),
            )

        for idx in range(self.mlp_depth):
            if self.mlp_type == "mlp":
                mlp.append(
                    FCBlock(self.mlp_size, self.mlp_size, dropout=self.dropout, bn=bn)
                )
            elif self.mlp_type == "res":
                mlp.append(ResBlock(dim=self.mlp_size, dropout=self.dropout, bn=bn))

        return mlp

    def get_FXParams(self, y: Tensor) -> Tensor:
        # ey = self.frontend(y)
        # ex = self.frontend(x)
        # e = ex-ey

        e = self.encoder(y)

        v_hat = self.sig(self.out_layer(e))
        return v_hat

    def forward(self, x: Tensor, y: Tensor):
        zhat = self.get_FXParams(y)
        yh = self.fx(x, zhat)
        return yh

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        # x = x / (x.std(2, keepdim=True) + 1e-3)
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        z = self.get_FXParams(y)
        y_hat = self.fx(x, z)

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()

        total_loss = self.audio_loss_weight * loss_audio
        total_loss = total_loss + self.params_loss_weight * loss_params

        self.log("loss_audio/train", loss_audio)
        self.log("loss_total/train", total_loss)
        self.log("loss_params/train", loss_params)
        self.log("params_std/train", z.std(0).mean(0))
        return total_loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        # x = x / (x.std(2, keepdim=True) + 1e-3)
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        z = self.get_FXParams(y)
        y_hat = self.fx(x, z)

        loss_audio = self.loss_fn(y_hat, y).mean()
        loss_params = self.param_loss_fn(v, z).mean()

        total_loss = self.audio_loss_weight * loss_audio
        total_loss = total_loss + self.params_loss_weight * loss_params

        self.log("loss_audio/valid", loss_audio)
        self.log("loss_total/valid", total_loss)
        self.log("loss_params/valid", loss_params)
        self.log("params_std/valid", z.std(0).mean(0))
        return total_loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        # x = x / (x.std(2, keepdim=True) + 1e-3)
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        z = self.get_FXParams(y)
        y_hat = self.fx(x, z)

        loss_params = self.param_loss_fn(v, z).mean()

        self.test_procedure(target=y, estimate=y_hat, name=["Best", "Estimated"])
        self.test_procedure(target=y, estimate=x, name="Input")
        self.log("Test/Estimated/Params", loss_params)
        self.log("Test/Best/Params", loss_params)


class Automatic_Masterer(FX_AE):
    def get_FXParams(self, x, y):
        e = self.frontend(x)
        v_hat = self.mlp(e)
        return v_hat


class FX_Inference(BEAFX):
    def __init__(
        self,
        fx: DDAFX,
        frontend_args: dict[str, str],
        mlp_depth: int,
        mlp_size: int,
        mlp_type: Literal["mlp", "res"] = "mlp",
        mlp_bn: bool = True,
        dropout: float = 0.0,
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
        num_mixtures: int = 1,
        distrib_type: Literal["full", "unif"] = "unif",
        base_entropy: Literal["direct", "MC"] = "direct",
        num_tries_best: int = 2,
        audio_loss_weight: float = 1.0,
        params_loss_weight: float = 0.0,
        optim_only_flow: bool = False,
    ):
        super().__init__(
            fx,
            audio_loss_fn,
            metrics_dict,
            learning_rate,
            lr_sched_patience,
            weight_decay,
        )
        self.save_hyperparameters(ignore=["metrics_dict", "loss_fn", "fx"])

        frontend = self.get_frontend(args=frontend_args)
        self.audio_loss_fn = audio_loss_fn

        params_dim = fx.num_parameters
        self.params_dim = params_dim

        self.start_beta = start_beta
        self.end_beta = end_beta
        self.beta = start_beta
        self.warmup_length = warmup_length

        self.num_tries_best = num_tries_best

        self.compute_base_entropy = base_entropy

        if context_size is None:
            self.get_c = lambda x: x
            context_size = mlp_size
        else:
            self.get_c = nn.Sequential(
                nn.Linear(mlp_size, context_size),
                # ResBlock(context_size),
                Swish(context_size),
            )

        self.flow_coupling = flow_coupling
        self.flow_layers_type = flow_layers_type

        self.context_size = context_size
        if self.flow_coupling:
            self.full_context_size = context_size + self.params_dim // 2
        else:
            self.full_context_size = context_size

        self.init_flow(
            flow_length=flow_length,
            flow_layers_type=flow_layers_type,
            flow_nl=flow_nl,
            flow_nl_knots=flow_nl_knots,
        )

        self.mlp_depth = mlp_depth
        self.mlp_size = mlp_size
        self.mlp_type = mlp_type
        self.mlp_bn = mlp_bn

        self.dropout = dropout

        mlp = self.init_MLP(frontend.out_dim)
        self.encoder = nn.Sequential(frontend, mlp)

        if num_mixtures == 1:
            # self.base_distrib: Distribution = OneDGaussian(
            #     params_dim, mlp_size, base_entropy=base_entropy
            # )
            if distrib_type == "gaussian_log":
                self.base_distrib: Distribution = Gaussian_LogSigma(
                    params_dim, mlp_size, base_entropy=base_entropy
                )
            else:
                self.base_distrib: Distribution = Gaussian(
                    params_dim, mlp_size, base_entropy=base_entropy
                )
            self.num_mixtures = 1
        elif distrib_type == "unif":
            self.base_distrib: Distribution = GMMUniform(
                params_dim,
                mlp_size,
                num_mixtures=num_mixtures,
                base_entropy=base_entropy,
            )
            self.num_mixtures = num_mixtures
        elif distrib_type == "full":
            self.base_distrib: Distribution = GMMFull(
                params_dim,
                mlp_size,
                num_mixtures=num_mixtures,
                base_entropy=base_entropy,
            )
            self.num_mixtures = num_mixtures

        self.optim_only_flow = optim_only_flow

    def get_weights_from_AE(self, ae: FX_AE):
        # ae = FX_AE.load_from_checkpoint(
        #     path, metrics_dict=self.metrics_dict, loss_fn=self.audio_loss_fn, fx=self.fx
        # )

        with torch.no_grad():
            self.encoder = ae.encoder

    def configure_optimizers(self):

        if self.optim_only_flow:
            optimizer = torch.optim.AdamW(
                list(self.base_distrib.parameters())
                + list(self.flow_layers.parameters())
                + list(self.get_c.parameters()),
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
                out_dict["monitor"] = "loss_total/valid"

            return out_dict
        else:
            return super().configure_optimizers()

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

    def init_flow(
        self,
        flow_length: int = 5,
        flow_layers_type: str = "static",
        flow_nl: str = "rq_spline",
        flow_nl_knots: int = 4,
    ):
        params_dim = self.params_dim
        context_size = self.context_size
        nl_bound = 8
        # self.flow_layers: nn.Sequential[flows.utils.Flow] = nn.Sequential()
        self.flow_layers = flows.utils.Flow_Sequential()
        # self.flow_layers: nn.Sequential[flows.utils.Flow] = nn.Sequential(flows.linear.Rectangular(in_dim=1, out_dim=params_dim))
        for layer_idx in range(flow_length):
            self.flow_layers.append(flows.linear.InvertPermutation(dim=params_dim))
            if flow_layers_type == "dynamic":
                # self.flow_layers.append(
                #     flows.linear.DynamicLower(
                #         dim=params_dim, context_size=context_size, offset=-1
                #     )
                # )
                # self.flow_layers.append(
                #     flows.linear.DynamicUpper(
                #         dim=params_dim, context_size=context_size, offset=0
                #     )
                # )
                self.flow_layers.append(
                    flows.linear.DynamicDiag(dim=params_dim, context_size=context_size)
                )
                self.flow_layers.append(
                    flows.linear.DynamicBias(dim=params_dim, context_size=context_size)
                )
            elif flow_layers_type == "static":
                self.flow_layers.append(flows.linear.StaticLower(dim=params_dim))
                self.flow_layers.append(
                    flows.linear.StaticUpper(dim=params_dim, offset=1)
                )

            if flow_nl == "rq_spline":
                self.flow_layers.append(
                    flows.splines.rq.Dynamic_AR(
                        dim=params_dim,
                        context_size=context_size,
                        num_knots=flow_nl_knots,
                        bound=nl_bound,
                    )
                )
            elif flow_nl == "dsf":
                self.flow_layers.append(
                    flows.DSF_Static(
                        dim=params_dim,
                        hidden_dim=flow_nl_knots,
                    )
                )
            elif flow_nl == "dsf_dyn":
                self.flow_layers.append(
                    flows.DSF_Dynamic(
                        dim=params_dim,
                        hidden_dim=flow_nl_knots,
                        context_size=context_size,
                    )
                )
            elif flow_nl == "iaf":
                self.flow_layers.append(flows.IAF_Static(dim=params_dim))
            elif flow_nl == "iaf_dyn":
                self.flow_layers.append(
                    flows.IAF_Dynamic(dim=params_dim, context_size=context_size)
                )
            elif flow_nl == "f++":
                self.flow_layers.append(
                    flows.FlowPP_Static(dim=params_dim, hidden_dim=flow_nl_knots)
                )
                self.flow_layers.append(
                    flows.IAF_Static(
                        dim=params_dim,
                    )
                )
            elif flow_nl == "f++_dyn":
                self.flow_layers.append(
                    flows.FlowPP_Dynamic(
                        dim=params_dim,
                        hidden_dim=flow_nl_knots,
                        context_size=context_size,
                    )
                )
                self.flow_layers.append(
                    flows.IAF_Dynamic(
                        dim=params_dim,
                        context_size=context_size,
                    )
                )
            else:
                if self.flow_coupling:
                    self.flow_layers.append(
                        flows.linear.CouplingBias(
                            dim=params_dim, context_size=context_size
                        )
                    )
                # else:
                #     self.flow_layers.append(
                #         flows.linear.DynamicBias(
                #             dim=params_dim, context_size=context_size
                #         )
                #     )
                if flow_nl == "res":
                    self.flow_layers.append(
                        flows.ParamLayer(flows.TanhLayer(), params_dim)
                    )
                if flow_nl == "tanh":
                    self.flow_layers.append(flows.TanhLayer())
                if flow_nl == "elu":
                    self.flow_layers.append(flows.ELULayer())

        if flow_length > 0 and flow_nl not in [
            "rq_spline",
            "dsf",
            "iaf_dyn",
            "iaf",
            "f++",
        ]:
            self.flow_layers.append(
                flows.linear.DynamicDiag(dim=params_dim, context_size=context_size)
            )
            self.flow_layers.append(
                flows.linear.DynamicBias(dim=params_dim, context_size=context_size)
            )

        self.flow_layers.append(flows.SigmoidLayer())

    def train_forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        d = self.params_dim

        embedding = self.encoder(y)
        c = self.get_c(embedding)

        z, H_base = self.base_distrib.sample_and_entropy(embedding)
        z0 = z.clone()
        H_flow = 0
        z, H_flow = self.flow_layers.forward_and_logdet(z, c)

        zT = z.clone()

        # latent_loss = -H0 - log_det

        v_hat = zT

        return v_hat, H_base.mean() / self.params_dim, H_flow.mean() / self.params_dim

    def forward(self, x, y) -> Tensor:
        # self.fx.to(self.device)

        bs = x.size(0)
        v_hat = self.get_FXParams(y=y)

        y_hat = self.fx(x, v_hat)
        return y_hat

    def get_FXParams(self, y: Tensor, num_samples: int = 1) -> Tensor:
        bs = y.size(0)
        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        if num_samples != 1:
            embedding = embedding.unsqueeze(1).expand(bs, num_samples, -1).flatten(0, 1)
            c = c.unsqueeze(1).expand(bs, num_samples, -1).flatten(0, 1)

        z, _ = self.base_distrib.sample_and_entropy(embedding)

        z: Tensor = self.flow_layers(z, c)

        if num_samples != 1:
            z = z.unflatten(0, (bs, num_samples))

        return z

    def get_FXParams_most_likely(
        self, y: Tensor, K: int = 2, return_logprob: bool = False
    ) -> Tensor:
        bs = y.size(0)

        mode_save = self.base_distrib.base_entropy
        self.base_distrib.base_entropy = "MC"

        embedding: Tensor = self.encoder(y)
        embedding = embedding.unsqueeze(1).expand(bs, K, -1).flatten(0, 1)
        c: Tensor = self.get_c(embedding)

        z, H_base = self.base_distrib.sample_and_entropy(embedding)

        z, H_flow = self.flow_layers.forward_and_logdet(z, c)

        log_prob = -H_base - H_flow
        log_prob = log_prob.reshape(bs, K)

        max_idx = (
            torch.argmax(log_prob, dim=1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(bs, 1, self.fx.num_parameters)
        )

        z = z.reshape(bs, K, self.fx.num_parameters)

        z = torch.gather(input=z, index=max_idx, dim=1).squeeze(1)

        self.base_distrib.base_entropy = mode_save
        if not return_logprob:
            return z
        else:
            return z, log_prob.amax(dim=1)

    def get_KLentropy_MMD_estimates(
        self, y: Tensor, K: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        bs = y.size(0)
        d = self.fx.num_parameters

        embedding: Tensor = self.encoder(y)
        embedding = embedding.unsqueeze(1).expand(bs, K, -1).flatten(0, 1)
        c: Tensor = self.get_c(embedding)

        z, H_base = self.base_distrib.sample_and_entropy(embedding)

        z, H_flow = self.flow_layers.forward_and_logdet(z, c)

        H_MC = (H_base + H_flow).unflatten(0, (bs, K)).mean(1)
        z = z.unflatten(0, (bs, K))

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
        mask = mask.unsqueeze(0)

        MMD = -2 * MMD_kernel(z.unsqueeze(1), z_prior.unsqueeze(2), C).mean((1, 2))

        temp = MMD_kernel(z.unsqueeze(1), z.unsqueeze(2), C)
        temp = temp * mask / (K * (K - 1))
        MMD = MMD + temp.sum((1, 2))

        temp = MMD_kernel(z_prior.unsqueeze(1), z_prior.unsqueeze(2), C)
        temp = temp * mask / (K * (K - 1))
        MMD = MMD + temp.sum((1, 2))

        return H_KL, MMD, H_MC

    def get_FXParams_and_entropy(self, y: Tensor, num_samples: int = 1) -> Tensor:
        bs = y.size(0)
        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        if num_samples != 1:
            embedding = embedding.unsqueeze(1).expand(bs, num_samples, -1).flatten(0, 1)
            c = c.unsqueeze(1).expand(bs, num_samples, -1).flatten(0, 1)

        z, H_base = self.base_distrib.sample_and_entropy(embedding)

        z, H_flow = self.flow_layers.forward_and_logdet(z, c)

        H = H_base + H_flow

        if num_samples != 1:
            z = z.unflatten(0, (bs, num_samples))
            H = H.unflatten(0, (bs, num_samples))

        return z, H

    def compute_losses(
        self, x: Tensor, y: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device
        # self.fx.to(device)
        bs = x.size(0)
        d = self.params_dim

        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        z_list, H_base, mix = self.base_distrib.sample_entropy_mixing(embedding)

        H_flow_tot = 0
        audio_loss_tot = 0
        params_loss_tot = 0

        num_mixtures = mix.size(1)

        for mix_idx in range(num_mixtures):
            z = z_list[:, mix_idx, :]
            z, H_flow = self.flow_layers.forward_and_logdet(z, c)
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
        # dim = self.params_dim

        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        z, H_base, mix = self.base_distrib.sample_entropy_mixing(embedding)

        K = mix.size(1)

        c = c.unsqueeze(1).expand(bs, K, self.context_size).flatten(0, 1)
        z = z.flatten(0, 1)

        zT, H_flow = self.flow_layers.forward_and_logdet(z, c)

        bs, _, N = x.size()
        x = x.expand(bs, K, N).flatten(0, 1).unsqueeze(1)
        y = y.expand(bs, K, N).flatten(0, 1).unsqueeze(1)

        y_hat = self.fx(x, zT)

        audio_loss = self.audio_loss_fn(y_hat, y).view(bs, K)
        audio_loss = (audio_loss * mix).sum(1).mean(0)

        H_flow = H_flow.view(bs, K)
        H_flow = (H_flow * mix).sum(1).mean()

        neg_entropy = -H_base.mean() - H_flow.mean()
        neg_entropy = neg_entropy / self.params_dim

        v = v.unsqueeze(1).expand(bs, K, v.size(1)).flatten(0, 1)
        params_loss = self.param_loss_fn(v, zT).view(bs, K)
        params_loss = (params_loss * mix).sum(1).mean(0)

        self.log("check/H_base", H_base.mean())
        self.log("check/H_flow", H_flow.mean())
        # self.log("check/mix/max", mix.amax(1).mean())
        # self.log("check/mix/min", mix.amin(1).mean())

        return audio_loss, neg_entropy, params_loss

    def compute_losses_optim_multiple(
        self, x: Tensor, y: Tensor, v: Tensor, num_iter: int = 1
    ) -> tuple[Tensor, Tensor, Tensor]:
        audio_loss = 0
        neg_entropy = 0
        params_loss = 0
        for i in range(num_iter):
            a, n, p = self.compute_losses_optim(x, y, v)
            audio_loss = audio_loss + a / num_iter
            neg_entropy = neg_entropy + n / num_iter
            params_loss = params_loss + p / num_iter
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
        # self.log("loss_params/train", params_loss)
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

        # Estimated loss computation

        z_1, H_base, H_flow = self.train_forward(x, y)
        y_hat_1 = self.fx(x, z_1)

        neg_entropy = -H_base - H_flow

        params_loss = self.param_loss_fn(v, z_1).mean()

        results_tensor = self.test_procedure(
            target=y, estimate=y_hat_1, name="Estimated"
        )
        self.test_procedure(target=y, estimate=x, name="Input")

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

        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

        embedding = (
            embedding.unsqueeze(1)
            .expand(bs, num_tries, embedding.size(1))
            .flatten(0, 1)
        )
        c = c.unsqueeze(1).expand(bs, num_tries, self.context_size).flatten(0, 1)

        z, _ = self.base_distrib.sample_and_entropy(embedding)
        # for layer in self.flow_layers:
        # z = layer(z, c)
        # z = self.out(z)
        z = self.flow_layers(z, c)

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

        self.test_procedure(target=y, estimate=y_best, name="Best")

        # Best params
        v_size = v.size(1)
        v = v.unsqueeze(1).expand(bs, num_tries, v_size).flatten(0, 1)
        params_loss = self.param_loss_fn(v, z)

        params_loss = params_loss.view(bs, num_tries)
        params_loss = params_loss.amin(1).mean()

        self.log("Test/Best/Params", params_loss)

        # Uniform best
        embedding: Tensor = self.encoder(y)
        c: Tensor = self.get_c(embedding)

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

    # def eval(self):
    #     super().eval()
    #     if isinstance(self.base_distrib, (GMMFull, GMMUniform)):
    #         self.base_distrib.base_entropy = "MC"
    #     return self

    # def train(self, mode=True):
    #     super().train(mode=mode)
    #     if isinstance(self.base_distrib, (GMMFull, GMMUniform)):
    #         if mode:
    #             self.base_distrib.base_entropy = self.compute_base_entropy
    #         else:
    #             self.base_distrib.base_entropy = "MC"
    #     return self


class SelfGen_AE(FX_AE):
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


class SelfGen_Inference(FX_Inference):
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


def get_model(
    model_name,
    fx: DDAFX,
    frontend_args: dict,
    metrics_dict: dict,
    loss_fn: nn.Module,
    mlp_depth: int,
    mlp_size: int,
    mlp_type: Literal["mlp", "res"] = "mlp",
    mlp_bn: bool = True,
    dropout: float = 0.0,
    num_mixtures: int = 1,
    distrib_type: Literal["full", "unif"] = "unif",
    base_entropy: Literal["direct", "MC"] = "direct",
    flow_length: int = 0,
    flow_layers_type: str = "static",
    flow_nl: str = "res",
    flow_nl_knots: int = 4,
    flow_coupling: bool = False,
    context_size: int = 32,
    start_beta: float = None,
    end_beta: float = 0.01,
    warmup_length: float = 50,
    learning_rate: float = 1e-4,
    lr_sched_patience: int = None,
    weight_decay: float = 3e-5,
    estimation_or_usage: Literal["estimation", "usage"] = "estimation",
    optim_only_flow: bool = False,
):
    if start_beta is None:
        start_beta = end_beta

    if model_name == "deter":
        model = FX_AE(
            frontend_args=frontend_args,
            fx=fx,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            mlp_bn=mlp_bn,
            dropout=dropout,
            metrics_dict=metrics_dict,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
        )
    elif model_name == "infer":
        model = FX_Inference(
            fx=fx,
            frontend_args=frontend_args,
            audio_loss_fn=loss_fn,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            mlp_bn=mlp_bn,
            dropout=dropout,
            start_beta=start_beta,
            end_beta=end_beta,
            warmup_length=warmup_length,
            metrics_dict=metrics_dict,
            flow_length=flow_length,
            flow_layers_type=flow_layers_type,
            flow_nl=flow_nl,
            flow_nl_knots=flow_nl_knots,
            flow_coupling=flow_coupling,
            context_size=context_size,
            learning_rate=learning_rate,
            distrib_type=distrib_type,
            base_entropy=base_entropy,
            num_mixtures=num_mixtures,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
            optim_only_flow=optim_only_flow,
        )
    else:
        raise ValueError(f"Wrong model type. You asked for '{model_name}'")

    return model


def get_model_selfgen(
    model_name,
    fx: DDAFX,
    frontend_args: dict,
    metrics_dict: dict,
    loss_fn: nn.Module,
    mlp_depth: int,
    mlp_size: int,
    mlp_type: Literal["mlp", "res"] = "mlp",
    mlp_bn: bool = True,
    dropout: float = 0.0,
    num_mixtures: int = 1,
    base_entropy: Literal["direct", "MC"] = "direct",
    distrib_type: Literal["full", "unif"] = "unif",
    flow_length: int = 0,
    flow_layers_type: str = "static",
    flow_nl: str = "res",
    flow_nl_knots: int = 4,
    flow_coupling: bool = False,
    context_size: int = 32,
    start_beta: float = None,
    end_beta: float = 0.01,
    warmup_length: float = 50,
    learning_rate: float = 1e-4,
    lr_sched_patience: int = None,
    weight_decay: float = 3e-5,
    estimation_or_usage: Literal["estimation", "usage"] = "estimation",
    audio_loss_weight=1.0,
    params_loss_weight=0.0,
    optim_only_flow: bool = False,
):
    if start_beta is None:
        start_beta = end_beta

    if model_name == "deter":
        model = SelfGen_AE(
            frontend_args=frontend_args,
            fx=fx,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            mlp_bn=mlp_bn,
            dropout=dropout,
            metrics_dict=metrics_dict,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
            audio_loss_weight=audio_loss_weight,
            params_loss_weight=params_loss_weight,
        )
    elif model_name == "infer":
        model = SelfGen_Inference(
            fx=fx,
            frontend_args=frontend_args,
            audio_loss_fn=loss_fn,
            mlp_depth=mlp_depth,
            mlp_size=mlp_size,
            mlp_type=mlp_type,
            mlp_bn=mlp_bn,
            dropout=dropout,
            start_beta=start_beta,
            end_beta=end_beta,
            warmup_length=warmup_length,
            metrics_dict=metrics_dict,
            flow_length=flow_length,
            flow_layers_type=flow_layers_type,
            flow_nl=flow_nl,
            flow_nl_knots=flow_nl_knots,
            flow_coupling=flow_coupling,
            context_size=context_size,
            learning_rate=learning_rate,
            base_entropy=base_entropy,
            distrib_type=distrib_type,
            num_mixtures=num_mixtures,
            lr_sched_patience=lr_sched_patience,
            weight_decay=weight_decay,
            audio_loss_weight=audio_loss_weight,
            params_loss_weight=params_loss_weight,
            num_tries_best=1,
        )
    else:
        raise ValueError(f"Wrong model type. You asked for '{model_name}'")

    return model
