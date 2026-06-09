import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy
import torchaudio
import numpy as np
import os
import pandas as pd
import sqlite3
from .main import DX7Preset
import io
from .ops_params import get_ops_params_idx
import random
from typing import Literal


class DX7_dataset_algo_56(Dataset):
    def __init__(
        self,
        sql_path="./AEAFX/data/dx7/dexed_presets.sqlite",
        samplerate: int = 44100,
        audio_length_s: float = 0.5,
        note_pitch: float = 261.63,  # C3 pitch
        random_permutations=True,
        ratio_normalization: Literal["lin", "log"] = "lin",
        noise_amp: float = 0.0,
    ):
        super().__init__()
        cnx = sqlite3.connect(sql_path, detect_types=sqlite3.PARSE_DECLTYPES)
        df = pd.read_sql_query("SELECT * FROM preset", cnx)
        all_presets_mat: np.ndarray = np.stack(df["pickled_params_np_array"].values)

        presets_list: list[DX7Preset] = []
        min_freq: float = 1
        max_freq: float = 1
        for idx in range(len(all_presets_mat)):
            m = io.BytesIO(all_presets_mat[idx])
            m.seek(0)
            m = np.load(m)

            preset = DX7Preset(m)

            if preset.algo == 5 or preset.algo == 6:

                if preset.condition():
                    ## Adding the presets in the list
                    presets_list.append(preset)
                    
                    ##Checking mininum and max preset frequencies
                    for op_idx in [1, 2, 3, 4, 5, 6]:
                        op_freq = preset.get_freq_op(op_idx)
                        min_freq = min(min_freq, op_freq)
                        max_freq = max(max_freq, op_freq)

        self.presets_list = presets_list
        self.min_freq_ratio = min_freq
        self.max_freq_ratio = max_freq
        self.random_permutations = random_permutations
        self.x = torch.zeros((1, int(samplerate * audio_length_s)))
        self.ratio_normalization = ratio_normalization
        self.noise_amp = noise_amp

    def __len__(self):
        return len(self.presets_list)

    @staticmethod
    def get_perm_idx():
        perm_identifier = random.randint(1, 6)

        if perm_identifier == 1:
            perm_indices = [0, 1, 2, 3, 4, 5]
        elif perm_identifier == 2:
            perm_indices = [0, 1, 4, 5, 2, 3]
        elif perm_identifier == 3:
            perm_indices = [2, 3, 0, 1, 4, 5]
        elif perm_identifier == 4:
            perm_indices = [4, 5, 0, 1, 2, 3]
        elif perm_identifier == 5:
            perm_indices = [2, 3, 4, 5, 0, 1]
        elif perm_identifier == 6:
            perm_indices = [4, 5, 2, 3, 0, 1]
        return perm_indices

    def norm_ratios(self, ratios: np.ndarray):
        M = self.max_freq_ratio
        m = self.min_freq_ratio
        if self.ratio_normalization == "lin":
            out = (ratios - m) / (M - m)
        if self.ratio_normalization == "log":
            out = (np.log(ratios) - np.log(m)) / (np.log(M) - np.log(m))

        return out

    def convert_preset(self, preset: DX7Preset):
        ratios = np.zeros(6)
        amps = np.zeros(6)

        all_ops_params_idx = get_ops_params_idx()

        for op_idx in [1, 2, 3, 4, 5, 6]:
            ops_params_idx = all_ops_params_idx[op_idx]

            ratios[op_idx - 1] = preset.get_freq_op(op_idx)
            amps[op_idx - 1] = preset.get_amp_op(op_idx)

        if self.random_permutations:
            # There are 3 equivalent bracnhes in the algorithm :
            # (1, 2), (3, 4) and (5, 6)
            perm_idx = self.get_perm_idx()

            ratios = ratios[perm_idx]
            amps = amps[perm_idx]

        amps = Tensor(amps)
        ratios = Tensor(self.norm_ratios(ratios))

        v = torch.cat((amps, ratios), dim=0)

        noise = torch.rand_like(v) - 0.5
        noise = noise * self.noise_amp

        v = v + noise

        v = torch.maximum(v, torch.zeros_like(v))
        v = torch.minimum(v, torch.ones_like(v))
        return v

    def __getitem__(self, index):

        preset: DX7Preset = self.presets_list[index]

        v = self.convert_preset(preset)

        return self.x, v


class DX7_dataset_algo_56_simple(Dataset):
    def __init__(
        self,
        sql_path="./AEAFX/data/dx7/dexed_presets.sqlite",
        samplerate: int = 44100,
        audio_length_s: float = 0.5,
        note_pitch: float = 261.63,  # C3 pitch
        random_permutations=True,
        ratio_normalization: Literal["lin", "log"] = "lin",
        noise_amp: float = 0.0,
        min_freq_arg: float = None,
        max_freq_arg: float = None,
    ):
        super().__init__()
        cnx = sqlite3.connect(sql_path, detect_types=sqlite3.PARSE_DECLTYPES)
        df = pd.read_sql_query("SELECT * FROM preset", cnx)
        all_presets_mat: np.ndarray = np.stack(df["pickled_params_np_array"].values)

        kept_idx = []
        presets_list: list[DX7Preset] = []
        min_freq: float = min_freq_arg
        max_freq: float = max_freq_arg

        self.min_freq_ratio = min_freq
        self.max_freq_ratio = max_freq
        self.ratio_normalization = ratio_normalization
        self.noise_amp = noise_amp

        bounded_min = min_freq_arg is not None
        bounded_max = max_freq_arg is not None

        for idx in range(len(all_presets_mat)):
            m = io.BytesIO(all_presets_mat[idx])
            m.seek(0)
            m = np.load(m)

            preset = DX7Preset(m)

            if preset.algo == 5 or preset.algo == 6:
                if preset.condition():
                    cond = True
                    for op in [1, 3, 5]:
                        if preset.get_freq_op(op) != 1 and preset.op_is_on(op):
                            cond = False
                    # Check if the preset min and max frequency ratios are inside the bounds
                    if bounded_min:
                        if preset.get_preset_min_freq() < min_freq:
                            cond = False
                    if bounded_max:
                        if preset.get_preset_max_freq() > max_freq:
                            cond = False

                    # Check if the preset is already in the list
                    for preset_from_list in presets_list:
                        if torch.all(
                            torch.eq(
                                self.convert_preset(preset),
                                self.convert_preset(preset_from_list),
                            )
                        ):
                            cond = False

                    if cond:
                        ## Adding the presets in the list
                        presets_list.append(preset)
                        kept_idx.append(idx)
                        ##Checking mininum and max preset frequencies
                        for op_idx in [1, 2, 3, 4, 5, 6]:
                            op_freq = preset.get_freq_op(op_idx)
                            min_freq = min(min_freq, op_freq)
                            max_freq = max(max_freq, op_freq)
        self.kept_idx = kept_idx

        self.presets_list = presets_list

        if bounded_min:
            self.min_freq_ratio = min_freq_arg
        else:
            self.min_freq_ratio = min_freq

        if bounded_max:
            self.max_freq_ratio = max_freq_arg
        else:
            self.max_freq_ratio = max_freq
        self.random_permutations = random_permutations
        self.x = torch.zeros((1, int(samplerate * audio_length_s)))

    def __len__(self):
        return len(self.presets_list)

    @staticmethod
    def get_perm_idx():
        perm_identifier = random.randint(1, 6)

        if perm_identifier == 1:
            perm_indices = [0, 1, 2, 3, 4, 5]
        elif perm_identifier == 2:
            perm_indices = [0, 1, 4, 5, 2, 3]
        elif perm_identifier == 3:
            perm_indices = [2, 3, 0, 1, 4, 5]
        elif perm_identifier == 4:
            perm_indices = [4, 5, 0, 1, 2, 3]
        elif perm_identifier == 5:
            perm_indices = [2, 3, 4, 5, 0, 1]
        elif perm_identifier == 6:
            perm_indices = [4, 5, 2, 3, 0, 1]
        return perm_indices

    def norm_ratios(self, ratios: np.ndarray):
        M = self.max_freq_ratio
        m = self.min_freq_ratio
        if self.ratio_normalization == "lin":
            out = (ratios - m) / (M - m)
        if self.ratio_normalization == "log":
            out = (np.log(ratios) - np.log(m)) / (np.log(M) - np.log(m))

        return out

    def convert_preset(self, preset: DX7Preset):
        ratios = np.zeros(3)
        amps = np.zeros(6)

        all_ops_params_idx = get_ops_params_idx()

        for op_idx in [2, 4, 6]:
            ratios[op_idx // 2 - 1] = preset.get_freq_op(op_idx)

        for op_idx in [1, 2, 3, 4, 5, 6]:
            ops_params_idx = all_ops_params_idx[op_idx]

            amps[op_idx - 1] = preset.get_amp_op(op_idx)

        amps = Tensor(amps)
        ratios = Tensor(self.norm_ratios(ratios))

        v = torch.cat((amps, ratios), dim=0)

        noise = torch.rand_like(v) - 0.5
        noise = noise * self.noise_amp

        v = v + noise

        v = torch.maximum(v, torch.zeros_like(v))
        v = torch.minimum(v, torch.ones_like(v))
        return v

    def __getitem__(self, index):

        preset: DX7Preset = self.presets_list[index]

        v = self.convert_preset(preset)

        return self.x, v


if __name__ == "__main__":
    ds = DX7_dataset_algo_56()
    print(ds.max_freq_ratio)
    print(ds.min_freq_ratio)
