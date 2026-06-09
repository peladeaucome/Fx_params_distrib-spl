ops_in_feedback: dict[list] = {
    1: [3, 4, 5, 6],
    2: [1, 2],
    3: [4, 5],
    4: [4, 5, 6],
    5: [5, 6],
    6: [5, 6],
    7: [3, 5, 6],
    8: [3, 4],
    9: [1, 2],
    10: [1, 2, 3],
    11: [4, 6],
    12: [1, 2],
    13: [3, 6],
    14: [3, 4, 6],
    15: [1, 2],
    16: [1, 5, 6],
    17: [1, 2],
    18: [1, 3],
    19: [4, 5, 6],
    20: [1, 2, 3],
    21: [1, 2, 3],
    22: [3, 4, 5, 6],
    23: [4, 5, 6],
    24: [3, 4, 5, 6],
    25: [4, 5, 6],
    26: [4, 6],
    27: [2, 3],
    28: [3, 4, 5],
    29: [5, 6],
    30: [3, 4, 5],
    31: [5, 6],
    32: [6],
}


def get_ops_in_fb() -> dict[int, list[int]]:
    """
    Returns a dictionnary with operator index as key and a list of the operators
    that are int the feedback loop as value
    """
    return ops_in_feedback
