"""
Polynomial-based, evolvable learning rule function based on pre-post spike dynamics

Based on 
Confavreux, B., Agnes, E. J., Zenke, F., Lillicrap, T., & Vogels, T. P. (2020). A meta-learning approach to (re)discover plasticity rules that carve a desired function into a neural network. Advances in Neural Information Processing Systems, 33, 16398–16408. https://doi.org/10.1101/2020.10.24.353409
Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. PLOS Computational Biology, 21(4), e1012910. https://doi.org/10.1371/journal.pcbi.1012910

"""

from typing import Literal, List, Tuple, Dict, Sequence
import numpy as np
from numpy.typing import ArrayLike

from common.base import LearningRule, Parameter
from lrule.base import BaseLearningRule
from genome.genome import EvolvableLearningRule


class SmallPolynomialRule(BaseLearningRule, EvolvableLearningRule):
    _new_specs = {
        "coefficients": dict(kind="real", length=4, dist="normal", dist_params=dict(loc=0, scale=1)),
    }
    default_gene_order = ("learning_rate", "tau_syn", "coefficients")

    def __init__(self, *, 
                parameters: ArrayLike = None, genes: List[Parameter] = None, 
                genes_to_encode: List[Dict] | Dict[str, Dict] = None, gene_order: Sequence[str] = None, 
                learning_rate: float = 1.0, learning_rate_thr: float = 0.1, threshold_agg_func: Literal["max", "min", "mean", "sum"] = "mean",
                delta_weight: bool = True, delta_threshold: bool = False, delta_eligibility: bool = False,
                # use_trace_pre: bool = False, use_trace_post: bool = False, use_spike_pre: bool = False, use_spike_post: bool = False,
                # use_weights: bool = True, use_reward: bool = False, 
                # use_eligibility: bool = False, use_eligibility_pre: bool = False, use_eligibility_post: bool = False, use_eligibility_stdp: bool = False,
                **kwargs
        ):
        if sum([bool(delta_weight), bool(delta_threshold), bool(delta_eligibility)]) != 1:
            raise ValueError("Only one output must be specified")

        BaseLearningRule.__init__(self, learning_rate=learning_rate, learning_rate_thr=learning_rate_thr, threshold_agg_func=threshold_agg_func, 
                                delta_weight=delta_weight, delta_threshold=delta_threshold, delta_eligibility=delta_eligibility,
                                use_trace_pre=True, use_trace_post=True, use_spike_pre=True, use_spike_post=True,
                                use_weights=False, use_reward=False, 
                                use_eligibility=False, use_eligibility_pre=False, 
                                use_eligibility_post=False, use_eligibility_stdp=False, 
                                **kwargs)

        EvolvableLearningRule.__init__(self, parameters=parameters, genes=genes, genes_to_encode=genes_to_encode, gene_order=gene_order)

        

    def _build_gene_specs(self):
        specs = super()._build_gene_specs()
        specs.update(self._new_specs)
        return specs

    def _apply_gene_values(self):
        super()._apply_gene_values()
        self.coefficients = self.values.get("coefficients")

    def forward(self, inp: np.ndarray) -> np.ndarray:
        spk_pre = inp[:, 2]
        spk_post = inp[:, 3]
        trace_pre = inp[:, 0]
        trace_post = inp[:, 1]
        coefs = self.coefficients
        dw = coefs[0] * spk_pre + coefs[1] * spk_post + coefs[2] * spk_post * trace_pre + coefs[3] * spk_pre * trace_post
        return dw

    @property
    def encode_coefficients(self) -> bool:
        return "coefficients" in self._gene_order
