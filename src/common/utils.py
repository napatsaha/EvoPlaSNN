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

def make_target_spikes(labels, num_classes=2):
    onehot = np.zeros((num_classes, len(labels)), dtype=np.int32)
    np.put_along_axis(onehot, np.array(labels).reshape((1, -1)), 1, axis=0)
    return onehot

def make_target_times(labels, num_clesses=2, buffer_length=10, max_value=None):
    if max_value is None:
        max_value = -buffer_length
    times = np.full((num_clesses, len(labels)), max_value)
    np.put_along_axis(times, np.array(labels).reshape((1, -1)), 0, axis=0)
    return times

def check_max_ties(array, axis=0):
    """
    Check if there are multiple maximum values in the array.
    Returns True if there are ties, False otherwise.
    """
    return np.sum(np.equal(array, np.max(array, axis=0)), axis=0) > 1