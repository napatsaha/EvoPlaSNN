from typing import Tuple
import numpy as np
from importlib import import_module

TYPE_DICT = {
    "ann" : ("lrule.ann", "ANN_Rule"),
    "cgp" : ("lrule.cgp", "CGP_Rule"),
    "graph" : ("lrule.cgp", "CGP_Graph"),
    "stdp" : ("lrule.stdp", "STDP_Rule"),
    "rstdp" : ("lrule.stdp", "R_STDP")
}



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


def create_learning_rule(type_: str = None, **kwargs):
    if type_ is None:
        if "type" in kwargs:
            type_ = kwargs.pop("type")
        else:
            raise KeyError("\'type\' must be provided in params")
    module_name, class_name = TYPE_DICT.get(type_)
    module = import_module(module_name)
    rule_cls = getattr(module, class_name)
    instance = rule_cls(**kwargs)
    return instance