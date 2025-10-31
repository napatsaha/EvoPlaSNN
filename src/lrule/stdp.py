from typing import Literal
import numpy as np
from abc import ABC, abstractmethod

# from snn.base import SynapseLayerProtocol
from .base import LearningRule
from .utils import tile_array


class STDP_Rule(LearningRule):
    def __init__(self, mu = 0.0, lambd = 0.1, alpha = 1.0, *,
                #  w_min: float = 0.0, w_max: float = 1.0, clip_w: bool = True,
                  condition: Literal["on-spike", "on-reward"] = "on-spike"):
        super().__init__()
        self.mu = mu
        self.lambd = lambd
        self.alpha = alpha
        # self.tau_trace = tau_trace
        # self.dt = dt
        # self.w_min = w_min
        # self.w_max = w_max
        # self.clip_w = clip_w
        assert condition in ("on-spike", "on-reward"), "Condition must be either 'on-spike' or 'on-reward'"
        self.condition = condition

    def F_neg(self, w):
        return self.lambd * self.alpha * (w**self.mu)
    
    def F_pos(self, w):
        return self.lambd * (1 - w)**self.mu

    def _stdp_update(self, w, spk_pre, spk_post, trace_pre, trace_post):
        """
        Update the weights based on the STDP rule.
        """
        # Compute the weight change
        if spk_post.any():
            # LTP = self.F_pos(w) * spk_post * trace_pre
            LTP = np.where(spk_post, self.F_pos(w) * trace_pre, 0)
        else:
            LTP = 0
        if spk_pre.any():
            # LTD = self.F_neg(w) * spk_pre * trace_post
            LTD = np.where(spk_pre, self.F_neg(w) * trace_post, 0)
        else:
            LTD = 0
        delta_w = LTP - LTD

        # delta_w = self.F_pos(w) * spk_post * trace_pre - self.F_neg(w) * spk_pre * trace_post
        return delta_w
        
    def update(self, synapse: 'SynapseLayerProtocol', reward=None) -> np.ndarray:

        w = synapse.weights
        w_shape = w.shape
        # Create tiled version of traces and spikes
        trace_pre, trace_post = tile_array(w_shape, synapse.pre_layer.get_trace(), synapse.post_layer.get_trace())
        spk_pre, spk_post = tile_array(w_shape, synapse.pre_layer.spike, synapse.post_layer.spike)

        dw = self._stdp_update(w, spk_pre, spk_post, trace_pre, trace_post)

        if reward is not None:
            dw = dw * reward


        # Update the weights
        # if self.condition == "on-spike":
        #     return dw
        # elif self.condition == "on-reward":
        #     if reward is None:
        #         return 0.0
        #     else:
        #         return dw * reward

        # Clip the weights
        # if self.clip_w:
        #     w = np.clip(w, self.w_min, self.w_max)
        return dw
    

class R_STDP(LearningRule):
    """
    An STDP-Rule that only uses Reward and Eligibility Trace as inputs.
    """
    def __init__(self, learning_rate: float = 0.01, use_eligibility_pre: bool = True, use_eligibility_post: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.use_eligibility_pre = use_eligibility_pre
        self.use_eligibility_post = use_eligibility_post
        self.delta_weight = True # For synapse.learning_rule.setter to register updating weights

    def update(self, synapse: 'SynapseLayerProtocol', reward=None, always_return_tuple: bool = False) -> np.ndarray:
        if reward is None:
            reward = 1.0

        # Get the eligibility trace
        if self.use_eligibility_pre:
            etrace_pre = synapse.eligibility_pre # shape: [N_pre, N_post]
        else:
            etrace_pre = 0.0

        if self.use_eligibility_post:
            etrace_post = synapse.eligibility_post # shape: [N_pre, N_post]
        else:
            etrace_post = 0.0

        # Combine the eligibility traces
        if self.use_eligibility_pre or self.use_eligibility_post:
            etrace = etrace_pre - etrace_post
        else:
            etrace = 1.0

        # Compute the weight change
        dw = self.learning_rate * reward * etrace

        if always_return_tuple:
            return dw, None
        else:
            return dw