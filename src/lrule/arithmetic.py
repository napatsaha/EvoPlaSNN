"""
Simple arithmetic-based rule. Originally meant to be used as the multiplication term of R-STDP in an external rule in Dual Learning Rule system.

Created: 2026-09-11

"""

from typing import Literal, List, Tuple, Dict, Sequence
import numpy as np
from numpy.typing import ArrayLike

from common.base import LearningRule, Parameter
from lrule.base import BaseLearningRule
from genome.genome import EvolvableLearningRule


class MultiplyLearningRule(BaseLearningRule, EvolvableLearningRule):
    """
    Simple product of all inputs
    """
    def __init__(self, *,
                parameters: ArrayLike = None, genes: List[Parameter] = None, 
                genes_to_encode: List[Dict] | Dict[str, Dict] = None, gene_order: Sequence[str] = None, 
                learning_rate: float = 1.0, learning_rate_thr: float = 0.1, threshold_agg_func: Literal["max", "min", "mean", "sum"] = "mean",
                trigger_condition: Literal["on-timestep", "on-step", "on-reward", "on-end"] = None,
                delta_weight: bool = False, delta_threshold: bool = False, delta_eligibility: bool = False,
                use_trace_pre: bool = False, use_trace_post: bool = False, use_spike_pre: bool = False, use_spike_post: bool = False,
                use_weights: bool = False, use_reward: bool = False, 
                use_eligibility: bool = False, use_eligibility_pre: bool = False, use_eligibility_post: bool = False, use_eligibility_stdp: bool = False,
                **kwargs
                 ):
        BaseLearningRule.__init__(self, learning_rate=learning_rate, learning_rate_thr=learning_rate_thr, threshold_agg_func=threshold_agg_func, 
                                  trigger_condition=trigger_condition, 
                                  delta_weight=delta_weight, delta_threshold=delta_threshold, delta_eligibility=delta_eligibility, 
                                  use_trace_pre=use_trace_pre, use_trace_post=use_trace_post, use_spike_pre=use_spike_pre, use_spike_post=use_spike_post, 
                                  use_weights=use_weights, use_reward=use_reward, 
                                  use_eligibility=use_eligibility, use_eligibility_pre=use_eligibility_pre, use_eligibility_post=use_eligibility_post, 
                                  use_eligibility_stdp=use_eligibility_stdp, **kwargs)
        
        EvolvableLearningRule.__init__(self, parameters=parameters, genes=genes, genes_to_encode=genes_to_encode, gene_order=gene_order)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        dw = np.prod(inp, axis=1)
        return dw
