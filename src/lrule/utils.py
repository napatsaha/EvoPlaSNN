from pathlib import Path
from typing import Tuple
from common.utils import _get_lrule_class
import numpy as np
from common.base import LearningRule

import yaml

def tile_array(target_shape: Tuple[int, int], vec_in: np.ndarray, vec_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def read_learning_rule(parameter_path: str | Path, config_path: str | Path) -> LearningRule:
    """
    Construct learning rule from a parameter ".txt" file and config ".yaml" file.

    The type is determined by the "type" key within "lrule_params" (default: ANN_Rule)
    """
    with open(config_path, 'r') as f:
        config: dict = yaml.safe_load(f)

    if "arule_params" in config:
        lrule_params = config.get("arule_params")
        lrule_type = "ann"
    elif "lrule_params" in config:
        lrule_params = config.get("lrule_params")
        lrule_type = lrule_params.pop("type")
        if lrule_type is None:
            raise ValueError("\'type\' not given in \'lrule_params\' of config file")
    else:
        raise ValueError("Either \'lrule_params\' or \'arule_params\' sub-dictionary must exist within config file.")
    
    parameters = np.loadtxt(parameter_path, delimiter=',')
    lrule_class = _get_lrule_class(lrule_type)

    return lrule_class(parameters=parameters, **lrule_params)

    