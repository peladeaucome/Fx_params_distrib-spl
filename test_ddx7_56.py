import numpy as np
import matplotlib.pyplot as plt
import torch
import AEAFX
from IPython.display import Audio
from torch import Tensor
import torch.utils.data
from train_ddx7_56 import (
    mel_loss,
    mfcc_loss,
    metrics_dict,
    OT_loss,
    DDX7Loss,
    OT_comp_loss,
    OT_RMS_loss,
    mrstft_revisited,
    get_model,
)
import lightning.pytorch as pl
import os
from tqdm import tqdm
import yaml

import matplotlib as mpl
import seaborn as sns
import pandas as pd
from scipy.special import gamma
import scipy.signal as sig
import scipy.io.wavfile
import nnAudio.features
from time import time
from tqdm import tqdm

if torch.cuda.is_available():
    accelerator = "gpu"
    device = "cuda:0"
else:
    accelerator = "cpu"
    device = "cpu"

print(device)

samplerate = 44100

two_f = AEAFX.loss.two_f_Model(samplerate=samplerate)


class NamedLoss:
    def __init__(self, loss, name: str, lowerisbetter: bool = True):
        self.loss = loss
        self.name = name
        self.lowerisbetter = lowerisbetter

    def __call__(self, x, y):
        return self.loss(x, y)


class NamedData:
    def __init__(self, data: Tensor, name: str):
        self.data = data
        self.name = name


def cosine_sim_mat(mat: Tensor):
    mat1 = mat.unsqueeze(0)
    mat2 = mat.unsqueeze(1)

    top = (mat1 * mat2).sum(2)
    bot = mat1.square().sum(2).sqrt() * mat2.square().sum(2).sqrt()
    return top / bot


def get_model_from_name(path: str, fx, device):
    if path == "random":
        model = RandomModel(fx, torch.nn.MSELoss())
    else:
        with open(path + ".yaml", "r") as f:
            cfg = yaml.safe_load(f)

        model = get_model(cfg, fx, fx)
        if device == "cpu":
            p = torch.load(path + ".pt", map_location=torch.device("cpu"))
        else:
            p = torch.load(
                path + ".pt",
            )

        model.load_state_dict(p, strict=False)
        if isinstance(model, AEAFX.models.FX_Inference):
            model.base_distrib.base_entropy = "MC"
    model = model.to(device).eval()
    return model


note_pitch = 110
samplerate = 44100
norm = "log"

ddx7 = AEAFX.ddafx.ddx7.DDX7_algo56_simple(
    base_freq=note_pitch,
    mod_max=12.57,
    min_ratio=0.5,
    max_ratio=10,
    samplerate=samplerate,
    ratio_norm=norm,
)


def nball_volume(d: int, R: float = 1.0):
    num = np.float_power(np.pi, d / 2) * np.power(R, d)
    den = gamma(d / 2 + 1)
    return num / den


def MMD_kernel(x: Tensor, y: Tensor, C: float = 1) -> Tensor:
    return C / (C + (x - y).square().sum(-1))


class RandomModel(AEAFX.models.BEAFX):
    def __init__(self, fx: AEAFX.ddafx.DDAFX, loss_fn):
        super().__init__(fx, loss_fn)
        self.fx = fx
        self.params_dim = self.fx.num_parameters

    def get_FXParams(self, y: Tensor, num_samples: int = 1):
        num_params = self.params_dim
        bs = y.size(0)

        if num_samples == 1:
            z = torch.rand((bs, num_params), device=y.device)
        else:
            z = torch.rand((bs, num_samples, num_params), device=y.device)

        return z

    def get_KLentropy_MMD_estimates(self, y, K):
        bs = y.size(0)
        d = self.params_dim

        z = self.get_FXParams(y, num_samples=K)

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


class NoModel(AEAFX.models.BEAFX):
    def __init__(self, fx: AEAFX.ddafx.DDAFX, loss_fn):
        super().__init__(fx, loss_fn)
        self.fx = fx
        self.params_dim = self.fx.num_parameters


def compute_losses(
    tested_model,
    test_loader,
    metrics_dict: dict,
    nbest: int = 1,
    device: torch.device = device,
):
    with torch.no_grad():
        tested_model = tested_model.to(device).eval()
        results_dict = {}
        H_MC_list = None
        H_KL_list = None
        MMD_list = None
        anaFx = tested_model.fx
        for i, batch in enumerate(iter(test_loader)):
            x, v = batch
            x: Tensor = x.to(device)
            v: Tensor = v.to(device)

            y: Tensor = anaFx(x, v)

            if isinstance(tested_model, AEAFX.models.FX_AE):
                tested_model: AEAFX.models.FX_AE
                z = tested_model.get_FXParams(y)
                yhat = anaFx(x, z)
                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.cpu().numpy()
                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full
            if isinstance(tested_model, AEAFX.models.FX_Inference):
                tested_model: AEAFX.models.FX_Inference

                bs = x.size(0)
                dim = tested_model.params_dim

                zT, entropy = tested_model.get_FXParams_and_entropy(
                    y, num_samples=nbest
                )
                if nbest != 1:
                    zT = zT.flatten(0, 1)

                bs, _, N = x.size()

                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)

                yhat = anaFx(x, zT)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                entropy = entropy.cpu().numpy()

                if H_MC_list is None:
                    H_MC_list = entropy
                else:
                    H_MC_list = np.concatenate((H_MC_list, entropy), axis=0)

                results_dict["H_MC"] = H_MC_list

                if nbest == 1:
                    H_KL, MMD, _ = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list

                    ### Computing results with the most likely params
                    bs = x.size(0)
                    zT = tested_model.get_FXParams_most_likely(y, K=10000)
                    yhat = anaFx(x, zT)

                    for m_key, metric in metrics_dict.items():
                        m_key = m_key + " most_likely"
                        loss_batch: Tensor = metric(y, yhat)

                        loss_batch = loss_batch.cpu().numpy()

                        if m_key in results_dict.keys():
                            loss_previous: np.ndarray = results_dict[m_key]
                            loss_full = np.concatenate(
                                (loss_batch, loss_previous), axis=0
                            )
                        else:
                            loss_full = loss_batch
                        results_dict[m_key] = loss_full

            if isinstance(tested_model, AEAFX.models.SynthPerm):
                tested_model: AEAFX.models.SynthPerm

                bs = x.size(0)
                dim = tested_model.params_dim

                z = tested_model.get_FXParams(y, num_steps=100, num_samples=nbest)
                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                if nbest > 1:
                    z = z.flatten(0, 1)
                yhat = tested_model.fx(x, z)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                if nbest == 1:
                    H_KL, MMD = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list

            if isinstance(tested_model, RandomModel):
                tested_model: RandomModel

                bs = x.size(0)
                dim = tested_model.params_dim

                z = tested_model.get_FXParams(y, num_samples=nbest)
                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                if nbest > 1:
                    z = z.flatten(0, 1)
                yhat = tested_model.fx(x, z)

                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full

                if nbest == 1:
                    H_KL, MMD = tested_model.get_KLentropy_MMD_estimates(y=y, K=100)
                    MMD = MMD.cpu().numpy()
                    H_KL = H_KL.cpu().numpy()
                    if MMD_list is None:
                        MMD_list = MMD
                    else:
                        MMD_list = np.concatenate((MMD_list, MMD), axis=0)
                    results_dict["MMD"] = MMD_list
                    if H_KL_list is None:
                        H_KL_list = H_KL
                    else:
                        H_KL_list = np.concatenate((H_KL_list, H_KL), axis=0)
                    results_dict["H_KL"] = H_KL_list
            if isinstance(tested_model, NoModel):
                tested_model: NoModel

                bs, _, N = x.size()
                x = x.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                yhat = x
                y = y.expand(bs, nbest, N).flatten(0, 1).unsqueeze(1)
                for m_key, metric in metrics_dict.items():
                    loss_batch: Tensor = metric(y, yhat)
                    loss_batch = loss_batch.unflatten(0, (bs, nbest)).amin(dim=1)

                    loss_batch = loss_batch.cpu().numpy()

                    if m_key in results_dict.keys():
                        loss_previous: np.ndarray = results_dict[m_key]
                        loss_full = np.concatenate((loss_batch, loss_previous), axis=0)
                    else:
                        loss_full = loss_batch
                    results_dict[m_key] = loss_full
    return results_dict


dir = "logs/ddx7_56/spl/"

metrics_dict = {
    "SOT": OT_comp_loss,
    "MR-STFT rev": mrstft_revisited,
    "neg SI-SDR": AEAFX.loss.neg_sisdr,
    "Mel": mel_loss,
    "MFCC": mfcc_loss,
    # "2f-model": two_f,
}

if "2f-model" in list(metrics_dict.keys()):
    output_dir = "test-spl-2fmodel"
    ds_mult = 2
else:
    ds_mult = 10
    output_dir = "test-spl"

# model_names_list = [
#     "mog-unif-6-ann",
#     "mog-unif-24-ann",
#     "synthperm-p2t",
# ]

model_names_list = [
    # "deter2",
    "gauss-1-ann",
    "gauss-2-ann",
    "mog-full-6-ann",
    "mog-unif-6-ann",
    "mog-unif-24-ann",
    # "gauss-1-noann",
    # "gauss-2-noann",
    # "mog-full-6-noann",
    # "mog-unif-6-noann",
    # "mog-unif-24-noann",
    # "synthperm-p2t",
    "synthperm-ffn",
    # "synthperm-p2t-ot",
    # "synthperm-ffn-ot",
    # "vae",
    # "mog-full-vae",
    "random"
]
save_tensor = True

# dataset_list = ["rand","lv"]
dataset_list = ["lv"]

# nbest_list = [1]
nbest_list = [10]
# nbest_list = [1]


for ds_name in dataset_list:
    print(ds_name)
    if ds_name == "rand":
        test_dataset = AEAFX.data.SelfGenDataset_DX7(
            audio_length=int(4 * samplerate),
            ds_length=1000 * ds_mult,
            num_params=9,
            num_amps=6,
            skew_amp_distrib=1,
        )
    elif ds_name == "lv":
        test_dataset = AEAFX.data.dx7.DX7_dataset_algo_56_simple(
            samplerate=samplerate,
            audio_length_s=4,
            note_pitch=note_pitch,
            ratio_normalization=norm,
            min_freq_arg=0.5,
            max_freq_arg=10,
        )
        # test_dataset = torch.utils.data.ConcatDataset(
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, num_workers=9
        )
        #     (test_dataset for i in range(10 * ds_mult))
        # )
    for model_name in model_names_list:
        print(model_name)
        for num_tries in nbest_list:
            if model_name == "random":
                model = RandomModel(ddx7,torch.nn.MSELoss())
            else:
                name = os.path.join(dir, model_name)
                model = get_model_from_name(name, ddx7, device)
            model = model.to(device)
            model = model.eval()

            model.num_tries_best = num_tries
            model.metrics_dict = metrics_dict
            results_dict = compute_losses(
                model, test_loader, metrics_dict, nbest=num_tries, device=device
            )

            for key in results_dict.keys():
                print(f"{key}: {results_dict[key].mean()}")
            print("\n")
            if save_tensor:
                save_dir = f"np_save/ddx7/{output_dir}/{ds_name}/{model_name}_nbest{model.num_tries_best}.npz"
                np.savez_compressed(save_dir, **results_dict)


print("End of script.")