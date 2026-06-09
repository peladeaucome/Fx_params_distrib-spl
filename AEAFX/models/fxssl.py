from . import main
import torch
from torch import Tensor


class FX_SSL(main.FX_AE):
    def __init__(
        self,
        frontend_args,
        fx,
        mlp_depth=2,
        mlp_size=64,
        mlp_type="mlp",
        mlp_bn=True,
        dropout=0,
        loss_fn=None,
        metrics_dict=None,
        learning_rate=0.001,
        lr_sched_patience=None,
        weight_decay=0.00003,
        audio_loss_weight=1,
        params_loss_weight=0,
    ):
        super().__init__(
            frontend_args,
            fx,
            mlp_depth,
            mlp_size,
            mlp_type,
            mlp_bn,
            dropout,
            loss_fn,
            metrics_dict,
            learning_rate,
            lr_sched_patience,
            weight_decay,
            audio_loss_weight,
            params_loss_weight,
        )
        self.rep_bias = None

    def set_representation_bias(self, list_batch: list[tuple[Tensor]]):

        rep_bias = 0
        with torch.no_grad():
            for batch in list_batch:
                x, _, _ = batch
                rep_bias += self.encoder[0](x.unsqueeze(0).to(self.device))
            rep_bias = rep_bias / len(list_batch)
            self.rep_bias = rep_bias

    def get_FXParams(self, y):
        e = self.encoder[0](y)
        if self.rep_bias is not None:
            e -= self.rep_bias
        z = self.sig(self.out_layer(self.encoder[1](e)))
        return z

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, v = batch
        # x = x / (x.std(2, keepdim=True) + 1e-3)
        # y = y / (y.std(2, keepdim=True) + 1e-3)
        ey: Tensor = self.encoder[0](y)
        ex: Tensor = self.encoder[0](x)

        loss_ssl_x = ex.square().mean()

        with torch.no_grad():
            loss_xy: Tensor = self.loss_fn(x, y)

        loss_ssl_y = (
            ((ey - ex).square().mean(1).sqrt() - loss_xy.sqrt()).square().mean()
        )

        ey = ey.detach()
        z = self.sig(self.out_layer(self.encoder[1](ey)))
        y_hat = self.fx(x, z)

        loss_ssl = loss_ssl_x + loss_ssl_y
        loss_audio = self.loss_fn(y_hat, y).mean()
        # loss_params = self.param_loss_fn(v, z).mean()

        total_loss = self.audio_loss_weight * loss_audio
        total_loss = total_loss + self.params_loss_weight * loss_ssl

        self.log("loss_audio/train", loss_audio)
        self.log("loss_total/train", total_loss)
        self.log("loss_ssl/train", loss_ssl)
        self.log("norm_ex/train", ex.square().mean(1).mean())
        self.log("norm_ey/train", ey.square().mean(1).mean())
        self.log("loss_xy/train", loss_xy.mean())
        self.log("loss_ssl_x/train", loss_ssl_x)
        self.log("loss_ssl_y/train", loss_ssl_y)
        self.log("params_std/train", z.std(0).mean(0))
        return total_loss
