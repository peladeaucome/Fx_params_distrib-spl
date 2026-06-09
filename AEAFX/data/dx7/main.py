import numpy as np
from .ops_in_fb_loop import get_ops_in_fb
from .ops_params import get_ops_params_idx
from typing import Union

all_ops_params = get_ops_params_idx()


class Operator:
    """
    DX7 Operator
    """

    def __init__(self, params_dict):
        self.params_dict = params_dict

    def render_sound(self, sr_hz, length_s, input=0):
        pass


class DX7Preset:
    """
    DX7 Preset. This class is made for easier processing.
    """

    def __init__(self, params: np.ndarray):
        self.params = params

        # Synthesis algorithm
        self.algo = round(params[4] * 31) + 1

        # operators in the feedback loop
        ops_in_fb_all = get_ops_in_fb()
        self.ops_in_fb = ops_in_fb_all[self.algo]

        # operators not in the feedback loop
        self.ops_not_in_fb = []
        for i in [1, 2, 3, 4, 5, 6]:
            if i not in self.ops_in_fb:
                self.ops_not_in_fb.append(i)

    def get_params(self):
        """
        Returns the preset normalized params
        """
        return self.params

    def op_is_on(self, op_idx: int):
        """
        Checks if an operator is on.
        """
        op_params_idx = all_ops_params[op_idx]
        if (
            self.params[op_params_idx["OUTPUT LEVEL"]] < 1e-3
            or self.params[op_params_idx["SWITCH"]] < 0.5
        ):
            on = False
        else:
            on = True
        return on

    def preset_uses_feedback(self) -> bool:
        """
        Checks whether the preset uses the feedback loop.
        """
        out = True

        global_fb = round(self.params[5] * 7)
        if global_fb == 0:
            out = False
        else:
            for op_idx in self.ops_in_fb:
                if not self.op_is_on(op_idx):
                    out = False

        return out

    def preset_uses_amp_modulation(self):
        global_mod = False
        op_mod = False

        for op_idx in range(1, 7):
            if self.op_uses_modulation(op_idx):
                op_mod = True
        if self.params[10] > 1e-3:
            global_mod = True

        if global_mod and op_mod:
            mod = True
        else:
            mod = False

        return mod

    def preset_uses_pitch_modulation(self):
        if self.params[9] > 1e-3 and self.params[14] > 1e-3:
            mod = True
        else:
            mod = False

        mod_env = False
        for i in range(19, 23):

            if (self.params[i] - 0.5) > 1e-3:
                mod_env = True

        mod = mod or mod_env
        return mod

    def op_uses_modulation(self, op_idx: int, eps=1e-2) -> bool:
        """
        Checks whether the operator is modulated by the LFO.
        """
        op_params_idx = all_ops_params[op_idx]
        mod_sens = self.params[op_params_idx["A MOD SENS."]]

        out = False
        if mod_sens > eps and self.op_is_on(op_idx):
            out = True
        return out

    def op_fixed_freq(self, op_idx: int) -> bool:
        """
        Checks whether the operator is modulated by the LFO.
        """
        op_params_idx = all_ops_params[op_idx]
        mode = self.params[op_params_idx["MODE"]]

        out = False
        if self.op_is_on(op_idx) and mode > 0.5:
            out = True
        return out

    def condition(self) -> bool:
        """
        Checks if the preset uses feedback,
        or modulation or fixed frequencies outside of the feedback loop.
        Returns FALSE if the preset USES feedback or fixed frequencies, TRUE otherwise.
        """
        fb = self.preset_uses_feedback()
        fixed_freq = False
        amp_mod = self.preset_uses_amp_modulation()
        pitch_mod = self.preset_uses_pitch_modulation()

        for op_idx in [1, 2, 3, 4, 5, 6]:
            if self.op_fixed_freq(op_idx) and self.op_is_on(op_idx):
                fixed_freq=True

        if fb or fixed_freq:
            out = False
        else:
            out = True

        return out

    def denorm_coarse(self, coarse_norm: float) -> float:
        coarse_idx = round(coarse_norm * 31)
        if coarse_idx == 0:
            coarse = 0.5
        else:
            coarse = coarse_idx
        return coarse

    def denorm_fine(self, fine_norm: float) -> float:
        fine_idx = round(fine_norm * 99)
        fine = fine_idx * 0.01
        return fine

    def get_freq_op(self, op_idx: int) -> float:
        """
        Computes the frequency ratio of an operator.
        We only use the coarse and fine frequencies as detune has very little
        effect : https://github.com/asb2m10/dexed/issues/88
        """
        op_params_idx_dict = all_ops_params[op_idx]

        coarse_norm = self.params[op_params_idx_dict["F COARSE"]]
        fine_norm = self.params[op_params_idx_dict["F FINE"]]
        detune_norm = self.params[op_params_idx_dict["F OSC DETUNE"]]

        coarse_idx = round(coarse_norm * 31)
        fine_idx = round(fine_norm * 99)
        detune_idx = round(detune_norm * 14) - 7

        if coarse_idx == 0:
            coarse = 0.5
        else:
            coarse = coarse_idx

        fine_mult = 1 + fine_idx * 0.01

        f_ratio = coarse * fine_mult
        return f_ratio

    def get_preset_max_freq(self):
        out = np.amin([self.get_freq_op(i) for i in range(1, 7)])
        for i in range(1, 7):
            if self.op_is_on(i):
                out = max(out, self.get_freq_op(i))
        return out

    def get_preset_min_freq(self):
        out = np.amax([self.get_freq_op(i) for i in range(1, 7)])
        for i in range(1, 7):
            if self.op_is_on(i):
                out = min(out, self.get_freq_op(i))
        return out

    def get_amp_op(self, op_idx: int) -> float:
        """
        Computes the frequency ratio of an operator.
        We only use the coarse and fine frequencies as detune has very little
        effect : https://github.com/asb2m10/dexed/issues/88
        """
        if self.op_is_on(op_idx):
            op_params_idx_dict = all_ops_params[op_idx]
            amp = self.params[op_params_idx_dict["OUTPUT LEVEL"]]
        else:
            amp=0

        return amp

    def __repr__(self):
        dict_params_names = {}
        out_str = ""
        for op_idx in [1, 2, 3, 4, 5, 6]:
            op_dict_params = all_ops_params[op_idx]
            for name, idx in op_dict_params.items():
                dict_params_names[idx] = name

        param_idx = 23
        for op_idx in [1, 2, 3, 4, 5, 6]:
            out_str = out_str + f"OP n {op_idx}\n\n"
            for i in range(22):
                name = dict_params_names[param_idx]
                p = self.params[param_idx]
                if name == "F COARSE":
                    p = self.denorm_coarse(p)
                elif name == "F FINE":
                    p = self.denorm_fine(p)
                elif name == "OUTPUT LEVEL":
                    p = round(p * 99)
                out_str += f"{name} : {p}\n"
                param_idx += 1
            out_str += "\n"
        return out_str

    def __eq__(self, other):
        if isinstance(other, DX7Preset):
            out=np.all((self.params == other.params))
        else:
            out = False
        return out

