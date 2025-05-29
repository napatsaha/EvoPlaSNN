import numpy as np


def solve_hidden(hidden_size):
    if isinstance(hidden_size, list | np.ndarray | tuple):
        if len(hidden_size) == 0:
            return []
        else:
            return [int(h) for h in hidden_size if h > 0]
    elif isinstance(hidden_size, int | float):
        if hidden_size <= 0 or np.isnan(hidden_size):
            return []
        else:
            return [int(hidden_size)]
    elif hidden_size is None:
        return []
    else:
        raise ValueError("hidden_size must be an int, float, list of ints, or None")
    

def get_layer_size(inp, out, bias):
    return (inp + int(bias)) * out


def calculate_size(inp, hid, out, bias):
    s = 0
    hid = solve_hidden(hid)
    sizes = [inp] + hid + [out]
    for l in range(len(sizes) - 1):
        s += get_layer_size(sizes[l], sizes[l + 1], bias)
    return s