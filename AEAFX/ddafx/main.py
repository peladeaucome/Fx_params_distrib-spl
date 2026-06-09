import torch
from torch import Tensor
from typing import Optional, Union, Literal
from ..data.fx.main import DAFx
import numpy as np


class DDAFX:
    def __init__(
        self,
        num_parameters: int,
        ranges_parameters: Tensor,
        samplerate: float = 44100,
    ):
        self.device = "cpu"
        self.samplerate = samplerate
        self.num_parameters: int = num_parameters
        self.ranges_parameters: Tensor = torch.tensor(
            ranges_parameters, device=self.device
        )
        self.mode: Literal["analysis", "synthesis"] = "analysis"

        self.name: str = "DDAFx"
        self.params_names: list[str]

    def process(self, x: Tensor, p: Tensor) -> Tensor:
        return x

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        eps = 1e-1
        v = self.denormalize_parameters(w)
        x = x - x.mean(2, keepdim=True)
        # x = x / (x.abs().amax(dim=2, keepdim=True) + eps)
        y = self.process(x, v)
        return y

    def __repr__(self):
        out = ""
        out += f"DDAFx: {self.num_parameters} parameters\n\n"

        out += f"{'Index':<8} | Parameter\n"
        out += f"{'-----':<8} -----------\n"

        for i in range(self.num_parameters):
            out += f"{i:<8} | {self.params_names[i]}\n"
        return out

    def denormalize_parameters(self, w: Tensor) -> Tensor:
        m = self.ranges_parameters[:, 0].reshape(1, self.num_parameters)
        M = self.ranges_parameters[:, 1].reshape(1, self.num_parameters)
        v = w * (M - m) + m
        return v

    def normalize_parameters(self, v: Tensor) -> Tensor:
        m = self.ranges_parameters[:, 0].reshape(1, self.num_parameters)
        M = self.ranges_parameters[:, 1].reshape(1, self.num_parameters)
        w = (v - m) / (M - m)
        return w

    def to(self, device: torch.device):
        self.ranges_parameters = self.ranges_parameters.to(device)
        self.device = device
        return self

    def apply(self, fn: callable):
        fn(self)
        fn(self.ranges_parameters)
        return self

    def cpu(self):
        self.ranges_parameters = self.ranges_parameters.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.ranges_parameters = self.ranges_parameters.cuda(device)
        return self

    def analysis(self):
        self.mode: str = "analysis"
        return self

    def synthesis(self):
        self.mode: str = "synthesis"
        return self

    def eval(self):
        self.train(False)
        return self

    def train(self, mode: bool):
        self.training = mode
        return self


class DDAFXChain(DDAFX):
    def __init__(self, *effects_args: DDAFX, samplerate: float = 44100):
        num_parameters = 0
        ranges_parameters = []

        self.effects_list: list[DDAFX] = []
        self.params_names = []

        for fx in effects_args:
            num_parameters += fx.num_parameters
            self.effects_list.append(fx)
            for i in range(fx.num_parameters):
                ranges_parameters.append(
                    [fx.ranges_parameters[i][0], fx.ranges_parameters[i][1]]
                )
                self.params_names.append(fx.params_names[i])

        super().__init__(num_parameters, ranges_parameters, samplerate)

    def __call__(self, x: Tensor, q: Tensor):
        out = x.clone()
        control_idx = 0

        for fx in self.effects_list:
            out = fx(out, q[:, control_idx : control_idx + fx.num_parameters])
            control_idx += fx.num_parameters
        return out

    def append(self, fx: DDAFX):
        self.effects_list.append(fx)
        self.num_parameters += fx.num_parameters
        self.ranges_parameters = torch.cat(
            (self.ranges_parameters, fx.ranges_parameters), dim=0
        )
        for i in range(fx.num_parameters):
            self.params_names.append(fx.params_names[i])

        return self
    
    def append_multiple(self, *fx_list):
        for fx in fx_list:
            self.append(fx)

    def to(self, device: torch.device):
        self.ranges_parameters = self.ranges_parameters.to(device)
        self.device = device
        for fx in self.effects_list:
            fx.to(device)
        return self

    def cpu(self):
        self.ranges_parameters = self.ranges_parameters.cpu()
        for fx in self.effects_list:
            fx.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.ranges_parameters = self.ranges_parameters.cuda(device)
        for fx in self.effects_list:
            fx.cuda(device)
        return self

    def __getitem__(self, idx):
        return self.effects_list[idx]

    def denormalize_parameters(self, w: Tensor) -> Tensor:

        v = torch.zeros_like(w)
        i = 0

        for fx in self.effects_list:
            fx_d = fx.num_parameters
            v[:, i : i + fx_d] = fx.denormalize_parameters(w[:, i : i + fx_d])
            i += fx_d
        return v


class DSPFX(DDAFX):
    def __init__(self, synFx: DAFx):
        num_params = synFx.num_parameters
        ranges_parameters = synFx.ranges_parameters
        samplerate = synFx.samplerate

        self.name="DSP"
        self.params_names = ["DSP param" for i in range(num_params)]

        super().__init__(
            num_parameters=num_params,
            ranges_parameters=ranges_parameters,
            samplerate=samplerate,
        )

        self.dsp_fx = synFx

    def process(self, x: Tensor, z: Tensor):
        device = x.device
        dtype = x.dtype

        if self.training:
            out = x
        else:
            with torch.no_grad():
                bs, C, N = x.size()
                d = z.size(1)

                x = x.cpu().numpy()
                z = z.cpu().numpy()

                out = np.zeros((bs, C, N))

                for batch_idx in range(bs):
                    out[batch_idx, 0] = self.dsp_fx(x[batch_idx, 0, :], z[batch_idx])

                out = torch.from_numpy(out).to(device).to(dtype)

        return out

    def __call__(self, x, w):
        return self.process(x, w)


class Volume(DDAFX):
    def __init__(
        self,
        min_volume_db: float = -20,
        max_volume_db: float = 20,
        samplerate: int = 44100,
    ):
        super().__init__(
            num_parameters=1,
            ranges_parameters=torch.tensor([[min_volume_db, max_volume_db]]),
            samplerate=samplerate,
        )

        self.params_names = ["Volume"]

    @staticmethod
    def idB20(x_dB: Tensor):
        x = torch.pow(10, x_dB * 0.05)
        return x

    def process(self, x, v):
        vol = self.idB20(v)
        x = x * (vol.unsqueeze(1))
        return x
