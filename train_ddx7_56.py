import AEAFX
import AEAFX.models as models
import torch
from torch import nn, Tensor
import torch.utils.data
import lightning.pytorch as pl
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os

torch.set_float32_matmul_precision("medium")

samplerate = 44100

if torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

if accelerator == "gpu":
    device = "cuda:0"
elif accelerator == "cpu":
    device = "cpu"


def get_FX(
    base_freq,
    mod_max: float = 2.0,
    min_ratio: float = 0.1,
    max_ratio: float = 10,
    samplerate: int = 44100,
    ratio_norm="lin",
):
    Fx = AEAFX.ddafx.ddx7.DDX7_algo56_simple(
        base_freq=base_freq,
        mod_max=mod_max,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        samplerate=samplerate,
        ratio_norm=ratio_norm,
    )
    return Fx


mrstft_revisited = AEAFX.loss.MR_STFT_Revisited().to(device)
mrstft_revisited_norm = AEAFX.loss.MR_STFT_Revisited_Norm().to(device)

norm_mel_loss = AEAFX.loss.NormalizedLogMel_Loss(
    sr=samplerate, n_fft=4096, win_length=None, n_mels=128, hop_length=1024
).to(device)

mel_loss = AEAFX.loss.LogMel_Loss(
    sr=samplerate, n_fft=4096, win_length=None, n_mels=128, hop_length=1024
).to(device)
mfcc_loss = AEAFX.loss.MFCC_Loss(
    sr=samplerate,
    n_fft=4096,
    win_length=None,
    n_mels=128,
    hop_length=1024,
    eps=1,
    num_features=128,
).to(device)

OT_loss = AEAFX.loss.SpectralOT_Loss(sr=samplerate, n_fft=64, hop_length=32).to(device)

OT_RMS_loss = AEAFX.loss.SpectralOT_RMS_Loss(
    sr=samplerate, n_fft=64, hop_length=32, weight_loud=0.2
).to(device)

OT_log_loss = AEAFX.loss.SpectralOT_Log_Loss(
    sr=samplerate, nbins=128, hop_length=512, fmin=100, fmax=4000
).to(device)


class DDX7Loss(nn.Module):
    def __init__(self, loss1, loss2, eps: float = 0.2):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor):
        out = self.loss1(x, y) + self.eps * self.loss2(x, y)
        return out


OT_comp_loss = DDX7Loss(OT_loss, mrstft_revisited, eps=0.05).to(device)
OT_log_comp_loss = DDX7Loss(OT_log_loss, mrstft_revisited, eps=0.05).to(device)


metrics_dict = {
    # "MR-STFT": mrstft,
    # "SOT" : OT_comp_loss,
    "MR-STFT rev": mrstft_revisited,
    # "MR-STFT rev-norm": mrstft_revisited_norm,
    "Mel": mel_loss,
    # "Mel-norm": norm_mel_loss,
    # "MSE": torch.nn.MSELoss(),
    "SI-SDR": AEAFX.loss.si_sdr,
    # "pimse": AEAFX.loss.pimse,
}


def get_model(
    cfg: dict,
    fx,
    audio_loss=None,
):
    model_name = cfg["model"]["name"]
    if model_name == "synthperm":
        model = AEAFX.models.SelfGen_Synthperm(
            fx=fx,
            frontend_args=cfg["model"]["frontend"],
            metrics_dict=metrics_dict,
            audio_loss_fn=audio_loss,
            mlp_depth=cfg["model"]["mlp"]["depth"],
            mlp_size=cfg["model"]["mlp"]["size"],
            mlp_type=cfg["model"]["mlp"]["type"],
            mlp_bn=cfg["model"]["mlp"]["bn"],
            vector_field_args=cfg["model"]["vector_field"],
            learning_rate=cfg["model"]["learning_rate"],
            minibatch_ot=cfg["model"]["minibatch_ot"],
        )
    else:
        model = AEAFX.models.get_model_selfgen(
            model_name=model_name.lower(),
            fx=fx,
            frontend_args=cfg["model"]["frontend"],
            metrics_dict=metrics_dict,
            loss_fn=audio_loss,
            audio_loss_weight=cfg["model"]["audio_loss_weight"],
            params_loss_weight=cfg["model"]["params_loss_weight"],
            start_beta=cfg["model"]["beta"]["start"],
            end_beta=cfg["model"]["beta"]["end"],
            context_size=cfg["model"]["flow"]["context_size"],
            mlp_depth=cfg["model"]["mlp"]["depth"],
            mlp_size=cfg["model"]["mlp"]["size"],
            mlp_type=cfg["model"]["mlp"]["type"],
            mlp_bn=cfg["model"]["mlp"]["bn"],
            flow_length=cfg["model"]["flow"]["length"],
            flow_layers_type=cfg["model"]["flow"]["layers"],
            flow_nl=cfg["model"]["flow"]["nl"]["name"],
            flow_nl_knots=cfg["model"]["flow"]["nl"]["knots"],
            flow_coupling=cfg["model"]["flow"]["coupling"],
            warmup_length=cfg["model"]["warmup_length"],
            num_mixtures=cfg["model"]["distrib"]["num_mixtures"],
            base_entropy=cfg["model"]["distrib"]["entropy"],
            distrib_type=cfg["model"]["distrib"]["type"],
            learning_rate=cfg["model"]["learning_rate"],
            lr_sched_patience=cfg["experiment"]["lr_sched_patience"],
            weight_decay=cfg["experiment"]["weight_decay"],
            estimation_or_usage=cfg["model"]["mode"],
            optim_only_flow=cfg["model"]["start_from_ae"],
        )
    return model


@hydra.main(version_base=None, config_path="conf", config_name="ddx7_56")
def main(cfg: DictConfig) -> None:
    verbose = True
    print(OmegaConf.to_yaml(cfg))
    print(HydraConfig.get().job.name)

    generator = torch.Generator().manual_seed(cfg["experiment"]["dataset_seed"])

    min_ratio = cfg["ddx7"]["min_ratio"]
    max_ratio = cfg["ddx7"]["max_ratio"]

    test_dataset = AEAFX.data.dx7.DX7_dataset_algo_56_simple(
        sql_path="AEAFX/data/dx7/dexed_presets.sqlite",
        samplerate=samplerate,
        audio_length_s=0.5,
        note_pitch=cfg["ddx7"]["base_freq"],
        random_permutations=True,
        ratio_normalization=cfg["ddx7"]["ratio_norm"],
        noise_amp=cfg["ddx7"]["noise_amp"],
        min_freq_arg=min_ratio,
        max_freq_arg=max_ratio,
    )

    # min_ratio = test_dataset.min_freq_ratio
    # max_ratio = test_dataset.max_freq_ratio

    # full_ds, _ = torch.utils.data.random_split(
    #     full_ds, lengths=[1, len(full_ds) - 1], generator=generator
    # )
    # full_ds =torch.utils.data.ConcatDataset((full_ds for _ in range(580)))

    bs = cfg["experiment"]["batch_size"]
    train_dataset = AEAFX.data.SelfGenDataset_DX7(
        audio_length=samplerate // 8,
        ds_length=int(bs * 4000),
        num_params=9,
        num_amps=6,
        skew_amp_distrib=1,
    )
    valid_dataset = AEAFX.data.SelfGenDataset_DX7(
        audio_length=samplerate // 8,
        ds_length=2048,
        num_params=9,
        num_amps=6,
        skew_amp_distrib=1,
    )

    fx = get_FX(
        base_freq=cfg["ddx7"]["base_freq"],
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        mod_max=cfg["ddx7"]["max_modulation_amplitude"],
        ratio_norm=cfg["ddx7"]["ratio_norm"],
    )

    if verbose:
        print("Getting loss function")

    # mrstft = AEAFX.loss.MR_STFT_Loss(
    #     n_ffts=[1024, 512, 256],
    #     hop_lengths=[512, 256, 128],
    #     window_sizes=[1024, 512, 256],
    #     samplerate=samplerate,
    # ).to(device)

    if verbose:
        print("Getting the neural network")

    model_name: str = cfg["model"]["name"]

    loss_str = cfg["model"]["loss"]

    if loss_str == "mel":
        audio_loss = mel_loss
    if loss_str == "mel_norm":
        audio_loss = norm_mel_loss
    if loss_str == "mrstft":
        audio_loss = mrstft_revisited
    if loss_str == "mfcc":
        audio_loss = mfcc_loss
    if loss_str == "sot":
        audio_loss = OT_comp_loss
    if loss_str == "sot_rms":
        audio_loss = OT_RMS_loss
    if loss_str == "sot_log":
        audio_loss = OT_log_comp_loss

    model=get_model(cfg, fx, audio_loss)

    if isinstance(model, AEAFX.models.FX_Inference):
        if cfg["model"]["start_from_ae"]:
            ae = AEAFX.models.FX_AE.load_from_checkpoint(
                "logs/ddx7_56/safe_sot/deter.ckpt",
                loss_fn=audio_loss,
                metrics_dict=metrics_dict,
                fx=fx,
            )
            model.get_weights_from_AE(ae)

    print(fx.ranges_parameters)
    if verbose:
        print("Getting datasets")

    test_dataset = torch.utils.data.ConcatDataset((test_dataset for _ in range(10)))
    if verbose:
        print("Getting dataloaders")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
    )

    if verbose:
        print("Setting the training")

    output_dir = HydraConfig.get().runtime.output_dir
    logger = pl.loggers.TensorBoardLogger(output_dir)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        filename="best",
        monitor="loss_total/valid",
        mode="min",
        save_last=True,
    )

    earlystop = EarlyStopping(
        monitor="loss_total/valid",
        mode="min",
        patience=cfg["experiment"]["early_stopping_patience"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, earlystop, lr_monitor]

    

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        deterministic=False,
        logger=logger,
        log_every_n_steps=50,
        max_epochs=cfg["experiment"]["max_epochs"],
        enable_progress_bar=cfg["experiment"]["progress_bar"],
        callbacks=callbacks,
        gradient_clip_val=1,
    )

    if verbose:
        print("Training")

    # model = torch.compile(model)

    if cfg["resume_training"] is None:
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
        )
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=cfg["resume_training"],
            weights_only=False
        )

    if verbose:
        print("Testing with the Le Vaillant's dataset")

    torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
    
    test_loss = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    test_loss = trainer.test(model, dataloaders=test_loader, ckpt_path="last")

    if verbose:
        print("Testing with the random dataset")

    test_ds = AEAFX.data.SelfGenDataset_DX7(
        audio_length=samplerate // 2, ds_length=10000, num_params=9, num_amps=6
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["experiment"]["batch_size"],
        num_workers=8,
        shuffle=False,
    )

    test_loss = trainer.test(model, dataloaders=test_loader, ckpt_path="best", weights_only=True)
    test_loss = trainer.test(model, dataloaders=test_loader, ckpt_path="last", weights_only=True)

    model = model.__class__.load_from_checkpoint(
        os.path.join(output_dir, "lightning_logs/version_0/checkpoints/last.ckpt"),
        weights_only=True,
        fx=fx,
        strict=True,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
    model = model.__class__.load_from_checkpoint(
        os.path.join(output_dir, "lightning_logs/version_0/checkpoints/best.ckpt"),
        weights_only=True,
        fx=fx,
        strict=True,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))


if __name__ == "__main__":
    main()
