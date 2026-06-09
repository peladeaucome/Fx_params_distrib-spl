ops_params = {
    6: {
        "EG RATE 1": 133,
        "EG RATE 2": 134,
        "EG RATE 3": 135,
        "EG RATE 4": 136,
        "EG LEVEL 1": 137,
        "EG LEVEL 2": 138,
        "EG LEVEL 3": 139,
        "EG LEVEL 4": 140,
        "OUTPUT LEVEL": 141,
        "MODE": 142,
        "F COARSE": 143,
        "F FINE": 144,
        "F OSC DETUNE": 145,
        "BREAK POINT": 146,
        "L SCALE DEPTH": 147,
        "R SCALE DEPTH": 148,
        "L KEYSCALE": 149,
        "R KEYSCALE": 150,
        "RATE SCALING": 151,
        "A MOD SENS.": 152,
        "KEY VELOCITY": 153,
        "SWITCH": 154,
    }
}

op6 = ops_params[6]
op_num_params = len(op6.keys())
list_keys = list(op6.keys())
param_idx = op6[list_keys[0]] - 1
for op_idx in range(5, 0, -1):
    op = {}
    for key in list_keys[::-1]:
        op[key] = param_idx
        param_idx += -1
    ops_params[op_idx] = op
    del op


def get_ops_params_idx() -> dict[int, dict[str, int]]:
    """
    Returns a dict with ints (operator indices) as key and a dict as value.
    The second dict has the parameter name as key and its index as value.
    This allows to know the index of a parameter given the operator and the parameter name.
    """
    return ops_params


if __name__ == "__main__":
    # print(ops_params)
    for k, v in ops_params.items():
        print(k, v, len(v))
