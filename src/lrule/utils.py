from typing import Tuple
import numpy as np



def tile_array(target_shape: Tuple[int, int], vec_in: np.ndarray, vec_out: np.ndarray) -> Tuple[np.ndarray]:
    """
    Reshape input and output vectors to match weight matrix.
    """
    if len(target_shape) != 2:
        raise ValueError("target_shape must be a tuple of length 2")
    if vec_in.shape[0] != target_shape[0]:
        raise ValueError(f"Size of input vector must match the first dimension of target_shape, got {vec_in.shape[0]} expected {target_shape[0]}")
    if vec_out.shape[0] != target_shape[1]:
        raise ValueError(f"Size of output vector must match the second dimension of target_shape, got {vec_out.shape[0]} expected {target_shape[1]}")

    vec_in = np.tile(vec_in, (target_shape[1], 1)).T
    vec_out = np.tile(vec_out, (target_shape[0], 1))
    return vec_in, vec_out