from typing import Literal, Union
import numpy as np
from common.base import LearningRule, SynapseLayerProtocol
from lrule.utils import tile_array


class Empty_Rule(LearningRule):
    """
    A dummy learning rule that does nothing.
    """
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs) -> float:
        # No update
        return 0.0
    

class BaseLearningRule(LearningRule):
    INPUT_ORDER = ("trace_pre", "trace_post", "weights", "reward", "eligibility_pre", "eligibility_post", "eligibility_stdp")
    OUTPUT_ORDER = ("weight", "threshold")
    AGG_DICT = {
                "max": np.max,
                "min": np.min,
                "mean": np.mean,
                "sum": np.sum
                }
    def __init__(self, parameters = None, *, 
                learning_rate: float = 1.0, learning_rate_thr: float = 0.1, threshold_agg_func: Literal["max", "min", "mean", "sum"] = "mean",
                delta_weight: bool = True, delta_threshold: bool = False,
                use_trace_pre: bool = False, use_trace_post: bool = False, use_weights: bool = True, use_reward: bool = False, 
                use_eligibility: bool = False, use_eligibility_pre: bool = False, use_eligibility_post: bool = False, use_eligibility_stdp: bool = False,
                **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.learning_rate_thr = learning_rate_thr
        if threshold_agg_func not in ("max", "min", "mean", "sum"):
            raise ValueError("threshold_agg_func must be one of 'max', 'min', 'mean', 'sum'")
        else:
            self.threshold_agg_func = threshold_agg_func
            self._thr_agg = self.AGG_DICT.get(threshold_agg_func)

        # Inputs to learning rule
        self.use_trace_pre = use_trace_pre
        self.use_trace_post = use_trace_post
        self.use_weights = use_weights
        self.use_reward = use_reward
        self.use_eligibility_pre = use_eligibility or use_eligibility_pre
        self.use_eligibility_post = use_eligibility_post
        self.use_eligibility_stdp = use_eligibility_stdp

        self.input_order = [item for item in self.INPUT_ORDER if getattr(self, f"use_{item}")]
        self.input_size = len(self.input_order)

        # Learning rule outputs
        self.delta_weight = delta_weight
        self.delta_threshold = delta_threshold
        if not (self.delta_weight or self.delta_threshold):
            raise ValueError("At least one of delta_weight or delta_threshold must be True.")
        self.output_size = int(self.delta_weight) + int(self.delta_threshold)
        self.output_order = [item for item in self.OUTPUT_ORDER if getattr(self, f"delta_{item}")]

    def prepare_inputs(self, synapse: SynapseLayerProtocol, reward: float, w_shape: tuple):
        inp = []
        # 1, 2 = trace pre, post
        if self.use_trace_pre or self.use_trace_post:
            trace_pre, trace_post = tile_array(w_shape, synapse.pre_layer.trace, synapse.post_layer.trace)
            if self.use_trace_pre:
                inp.append(trace_pre.reshape(-1, 1))
            if self.use_trace_post:
                inp.append(trace_post.reshape(-1, 1))
        # 3 = weights
        if self.use_weights:
            inp.append(synapse.weights.reshape(-1, 1))
        # 4 = reward
        if self.use_reward:
            if reward is None:
                reward = 0
            inp.append(np.full((np.prod(w_shape), 1), fill_value=reward))
        # 5 = etrace pre-before-post (LTP)
        if self.use_eligibility_pre:
            inp.append(synapse.eligibility_pre.reshape(-1, 1))
        # 6 = etrace post-before-pre (LTD)
        if self.use_eligibility_post:
            inp.append(synapse.eligibility_post.reshape(-1, 1))
        # 7 = etrace STDP
        if self.use_eligibility_stdp:
            inp.append(synapse.eligibility_stdp.reshape(-1, 1))

        inp = np.concatenate(inp, axis=1)
        return inp
        
    def prepare_output(self, out: np.ndarray, always_return_tuple: bool = False) -> Union[tuple, np.ndarray]:
        if self.delta_weight:
            idx = 0
            dw = out[..., idx]
            dw *= self.learning_rate
        if self.delta_threshold:
            idx = 1 if self.delta_weight else 0
            dth = out[..., idx]
            dth = self._thr_agg(dth, axis=0) # Aggregate threshold deltas for each post-synaptic neuron
            dth *= self.learning_rate_thr

        # Return values
        if self.delta_weight and self.delta_threshold:
            return dw, dth
        elif self.delta_weight and not self.delta_threshold:
            if always_return_tuple:
                return dw, None
            else:
                return dw
        elif self.delta_threshold and not self.delta_weight:
            if always_return_tuple:
                return None, dth
            else:
                return dth
        else:
            raise RuntimeError("At least one of delta_weight or delta_threshold must be True.")
        
    def update(self, synapse: SynapseLayerProtocol, reward: float = None, always_return_tuple: bool = False) -> np.ndarray: 
        """
        Apply the ANN Rule to an external set of weights.
        """
        w_shape = synapse.weights.shape

        inp = self.prepare_inputs(synapse, reward, w_shape)
        out = self.forward(inp)
        out = out.reshape(*w_shape, -1)

        return self.prepare_output(out, always_return_tuple=always_return_tuple)
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Each Learning Rule needs to implement its own forward method")
    
    @property
    def size(self):
        raise NotImplementedError("Subclasses should implement method to return genome size")
    
    @property
    def parameters(self):
        raise NotImplementedError("Subclasses should implement getter and setter for internal parameters")
    