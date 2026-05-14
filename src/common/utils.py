from typing import List, Tuple
from importlib import import_module
import warnings

import numpy as np

from common.base import Solver, LearningRule

## HELPER FUNCTIONS

def solve_hidden(hidden_size) -> List[int]:
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

def safe_check_length(a, b) -> bool:
    """
    A safe way to check if the length of two objects are equal

    Args:
        a (Any): Object A
        b (Any): Object B

    Returns:
        bool: Whether or not the two objects have the same length
    """
    pass

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


def compare_deep_dict(d1: dict, d2: dict, warn_missing_keys: bool = True):
    """
    Compare differences between two deep dictionaries. Fields which exist in one
    dict but not the other will be included but the value for that field left as None.
    Returns a nested dictionary with a tuple of values from each dictionary which are
    different.

    Args:
        d1 (dict): First nested dictionary
        d2 (dict): Second nested dictionary
        warn_missing_keys (bool, optional): Whether or not to issue a warning when
        encountering a field that exist in only one dict but not the other. Defaults to True.

    Raises:
        Warning: When a mismatched field between two dicts is encountered, if 
        warn_missing_keys is enabled

    Returns:
        Dict[str: Tuple]: A nested dictionary of differences
    """
    diffs = {}
    unique_keys = set([*d1.keys(), *d2.keys()])
    for key in unique_keys:
        item1 = d1.get(key, None)
        item2 = d2.get(key, None)
        if (key in d1.keys()) and (key in d2.keys()):
            # item1 = d1.get(key)
            # item2 = d2.get(key)
            is_dict1 = isinstance(item1, dict)
            is_dict2 = isinstance(item2, dict)
            if is_dict1 and is_dict2:
                is_diff = compare_deep_dict(item1, item2, warn_missing_keys=warn_missing_keys)
                if is_diff:
                    diffs[key] = is_diff
            elif (is_dict1 and not is_dict2) or (is_dict2 and not is_dict1):
                warnings.warn(f"Found key {key} common between two dictionaries, but Dict1 {is_dict1} while Dict2 {is_dict2}. Cannot handle such case.")
            else:
                if item1 == item2:
                    continue
                else:
                    diffs[key] = (item1, item2)
        elif (key in d1.keys()) and (key not in d2.keys()):
            if warn_missing_keys:
                warnings.warn(f"Found key: {key} in Dict1 but not in Dict2")
            diffs[key] = (item1, item2)
        elif (key not in d1.keys()) and (key in d2.keys()):
            if warn_missing_keys:
                warnings.warn(f"Found key: {key} in Dict2 but not in Dict1")
            diffs[key] = (item1, item2)
        else:
            raise Warning("Reached unsupported case where key not in either Dict1 or Dict2")
    return diffs


def get_boundaries_for_lrule_inputs(simulator: 'SNNSimulator', input_var: str) -> Tuple[float, float]:
    _supported_inputs = ("trace_pre", "trace_post", "weights", "reward", "eligibility_pre", "eligibility_post", "eligibility_stdp")
    assert input_var in _supported_inputs, f"Input variable needs to be one of {_supported_inputs}. Got {input_var}"
    min_, max_ = None, None
    try:
        if input_var == "reward":
            rlist = simulator.env.reward_list
            min_ = min(rlist)
            max_ = max(rlist)
        elif input_var == "weights":
            if simulator.network.synapse_layers[0].clip_weights:
                min_ = simulator.network.synapse_layers[0].weight_clip_min
                max_ = simulator.network.synapse_layers[0].weight_clip_max
            elif simulator.weight_recorder is not None:
                min_ = simulator.weight_recorder.values[0].min()
                max_ = simulator.weight_recorder.values[0].max()
        elif input_var == "eligibility_pre":
            min_ = 0
            if simulator.eligibility_pre_recorder is not None:
                max_ = simulator.eligibility_pre_recorder.values[0].max()
            else:
                max_ = simulator.network.synapse_layers[0].e_max
        elif input_var == "eligibility_post":
            min_ = 0
            if simulator.eligibility_post_recorder is not None:
                max_ = simulator.eligibility_post_recorder.values[0].max()
            else:
                max_ = simulator.network.synapse_layers[0].e_max
        elif input_var == "eligibility_stdp":
            if simulator.eligibility_stdp_recorder is not None:
                max_ = simulator.eligibility_stdp_recorder.values[0].max()
                min_ = simulator.eligibility_stdp_recorder.values[0].min()
            else:
                max_ = simulator.network.synapse_layers[0].e_max
                min_ = None
        else:
            raise NotImplementedError(f"Method for finding bounds for variable {input_var} not yet implemented.")
    except Exception as exc:
        warnings.warn(f"Could not find bounds for {input_var}. Using default min={min_}, max={max_}")
        warnings.warn(f"Got exception: {exc}")
    return min_, max_

def assymetric_min_max_normalise(a, center=0):
    upper_bound = np.max(a)
    lower_bound = np.min(a)
    upper_range = upper_bound - center
    lower_range = center - lower_bound
    return np.where(a >= center, (a - center) / upper_range, (a - center) / lower_range)

def make_input_grid(bounds, N):
    xii = [np.linspace(low, upp, N) for low, upp in bounds]
    xgrids = np.meshgrid(*xii)
    inp = np.concatenate([xi.reshape(-1, 1) for xi in xgrids], axis=-1)
    return inp


## Factory Methods

ALGO_DICT = {
    "cma_es": ("evo.cma_es", "CMA_ES"),
    "es": ("evo.es", "EvolutionStrategy"),
    "simple": ("evo.es", "EvolutionStrategy"),
    "mu_plus_lambda": ("evo.mu_lambda", "MuPlusLambda")
}


def create_solver(params: dict, **kwargs) -> Solver:
    params = params.copy()
    if "type" in params:
        solver_type = params.pop("type")
    else:
        raise Warning("Solver type not provided. Using \'Simple ES\'")
    solver_type = solver_type.lower().replace("-", "_")
    module_name, class_name = ALGO_DICT.get(solver_type)
    module = import_module(module_name)
    obj_class = getattr(module, class_name)
    return obj_class(**params, **kwargs)


TYPE_DICT = {
    "ann" : ("lrule.ann", "ANN_Rule"),
    "cgp" : ("lrule.cgp", "CGP_Rule"),
    "graph" : ("lrule.cgp", "CGP_Graph"),
    # "stdp" : ("lrule.stdp", "STDP_Rule"),
    "rstdp" : ("lrule.stdp", "R_STDP_Rule")
}


def _get_lrule_class(lrule_type):
    module_name, class_name = TYPE_DICT.get(lrule_type)
    module = import_module(module_name)
    rule_cls = getattr(module, class_name)
    return rule_cls


def create_learning_rule(type_: str = None, **kwargs) -> LearningRule:
    """
    Factory method for creating instance of a Learning Rule object, based on `type`
    """
    if type_ is None:
        if "type" in kwargs:
            type_ = kwargs.pop("type")
        elif "name" in kwargs:
            type_ = kwargs.pop("name")
        else:
            raise KeyError("\'type\' must be provided in params")
    rule_cls = _get_lrule_class(type_)
    instance = rule_cls(**kwargs)
    return instance


