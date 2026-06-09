import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube


class SinesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        freqs: tuple = (1000, 3000),
        length_s=1,
        samplerate: float = 16000,
        length=1000,
    ):
        super().__init__()
        self.length = length

        self.num_components: int = len(freqs)

        length_sp = int(length_s * samplerate)

        n = torch.arange(length_sp).reshape(1, length_sp, 1)
        t = n / samplerate

        freqs_tensor = torch.Tensor(list(freqs)).reshape(1, 1, self.num_components)

        # Prepare the sines
        self.sines = torch.sin(t * freqs_tensor * 2 * np.pi)

        self.amplitudes = torch.rand((length, 1, 1, self.num_components)) * 2 - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        v = self.amplitudes[idx]
        y = (self.sines * v).sum(2)
        v = v.reshape(self.num_components)
        x = torch.zeros_like(y)
        return x, y, v


# class SelfGenDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, audio_length: int = 44100, ds_length: int = 100, num_params: int = 12
#     ):
#         super().__init__()
#         self.x = torch.zeros((1, audio_length))
#         self.ds_length = ds_length

#         sampler = LatinHypercube(d=num_params)
#         self.v_list = sampler.random(ds_length)

#     def __len__(self):
#         return self.ds_length

#     def __getitem__(self, idx):

#         v = torch.Tensor(self.v_list[idx])
#         x = self.x

#         return x, v


class SelfGenDataset(torch.utils.data.Dataset):
    def __init__(
        self, audio_length: int = 44100, ds_length: int = 100, num_params: int = 12
    ):
        super().__init__()
        self.x = torch.zeros((1, audio_length))
        self.ds_length = ds_length

        self.num_params = num_params

    def __len__(self):
        return self.ds_length

    def __getitem__(self, idx):

        v = torch.rand(self.num_params)
        x = self.x

        return x, v


class SelfGenDataset_DX7(SelfGenDataset):
    def __init__(
        self,
        audio_length: int = 44100,
        ds_length: int = 100,
        num_params: int = 2,
        num_amps=1,
        skew_amp_distrib: int = 1,
    ):
        super().__init__(
            audio_length=audio_length, ds_length=ds_length, num_params=num_params
        )
        self.num_amps = num_amps
        self.skew_amp_distrib = skew_amp_distrib

    def __getitem__(self, idx):
        x, v = super().__getitem__(idx)

        if self.num_amps != 0:
            for i in range(self.skew_amp_distrib):
                v[: self.num_amps] = torch.log(v[: self.num_amps] * np.expm1(1) + 1)
        return x, v
