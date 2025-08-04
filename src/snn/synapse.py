from typing import Literal
import numpy as np
# from .neurons import NeuronLayer
from .base import SynapseLayerProtocol, NeuronLayerProtocol
from lrule import ANN_Rule, LearningRule, Empty_Rule, STDP_Rule
# from .utils import tile

class SynapseLayer(SynapseLayerProtocol):
    """
    Purpose: To represent a collection of synapse that connects one layer of neurons to another.
    (Assuming a Sequential Linear Layer network with all-to-all connections)

    Functionality:
    `forward()`: computes the output current to next neuron layer given a spike current input 
    from previous neuron layer -- i.e. weighted spike current
    `update()`: Update the entire synaptic weight efficacies based on a particular LearningRule
    `reset()`: Reset the synaptic weights to their initial state.
    `eligibility_trace`: If using eligibility traces, return the eligibility trace.
    `update_eligibility_trace()`: Update the eligibility trace based on the pre and post neuron layer spikes.
    """
    post_layer: NeuronLayerProtocol
    pre_layer: NeuronLayerProtocol
    weights: np.ndarray
    learning_rule: LearningRule
    
    def __init__(self, pre_layer: NeuronLayerProtocol, post_layer: NeuronLayerProtocol, *, 
                 learning_rule: LearningRule = None,
                 eligibility_trace: bool = False, tau_syn: float = None, dt: float = 1e-3,
                 weight_init: str = 'uniform', weight_init_params: dict = None, 
                 weight_min: float = 0.0, weight_max: float = 1.0,
                 clip_weights: bool = True, normalise_weights: bool = False, 
                 normalise_method: Literal["sum", "L2", "P"] = "sum", normalise_params: dict = None):

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()
        self._learning_rule_type = self._get_lrule_type()
        self._use_elig = eligibility_trace
        if self._use_elig:
            self._etrace = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
        if tau_syn is not None and isinstance(tau_syn, int):
            tau_syn = tau_syn * dt
        self.tau_syn = tau_syn if tau_syn is not None else dt
        self.dt = dt

        # Initialize weights
        self.weight_init = weight_init
        self.weight_init_params = weight_init_params
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.clip_weights = clip_weights
        self.normalise_weights = normalise_weights
        self.normalise_method = normalise_method
        self.normalise_params = normalise_params if normalise_params is not None else {}
        self._init_weights()

    def _init_weights(self):
        self.weights = np.random.uniform(self.weight_min, self.weight_max, size=(self.pre_layer.size, self.post_layer.size))
        # if self.weight_init_params is None:
        #     self.weight_init_params = {}
        # if self.weight_init == 'uniform':
        #     low = self.weight_init_params.get('low', 0.0)
        #     high = self.weight_init_params.get('high', 1.0)
        #     self.weights = np.random.uniform(low, high, size=(self.pre_layer.size, self.post_layer.size))
        # else:
        #     raise NotImplementedError("Other weight initialisation methods not implemented yet")

        self._normalise_weights()

    def _get_lrule_type(self):
        if isinstance(self.learning_rule, STDP_Rule):
            return "STDP"
        elif isinstance(self.learning_rule, ANN_Rule):
            return "ANN"
        elif isinstance(self.learning_rule, Empty_Rule):
            return None

    def _tile(self, vec_in: np.ndarray, vec_out: np.ndarray) -> np.ndarray:
        """
        Reshape input and output vectors to match weight matrix.
        """
        assert vec_in.shape[0] == self.pre_layer.size
        assert vec_out.shape[0] == self.post_layer.size
        vec_in = np.tile(vec_in, (self.weights.shape[1], 1)).T
        vec_out = np.tile(vec_out, (self.weights.shape[0], 1))
        return vec_in, vec_out
    
    def reset(self) -> None:
        """
        Reset the synaptic weights to their initial state.
        """
        # self._init_weights()
        self.weights[:] = np.random.uniform(self.weight_min, self.weight_max, size=(self.pre_layer.size, self.post_layer.size))
        self._normalise_weights()
        if self._use_elig:
            self._etrace.fill(0.0)

    def soft_reset(self) -> None:
        """
        Reset only eligibility traces, but keep the synaptic weights unchanged.
        """
        if self._use_elig:
            self._etrace.fill(0.0)

    def forward(self, spike_input: np.ndarray) -> np.ndarray:
        """
        Compute the output current to the next neuron layer given a spike current input from the previous neuron layer.
        """
        assert spike_input.shape[0] == self.pre_layer.size
        assert spike_input.ndim == 1

        # Compute the output current
        output_current = np.dot(spike_input, self.weights)
        return output_current
    
    def update_eligibility_trace(self) -> None:
        if self._use_elig:
            post_spike = self.post_layer.spike
            if sum(post_spike) == 0:
                rise = 0.0
            else:        
                pre_trace = self.pre_layer.get_trace()
                pre_trace, post_spike = self._tile(pre_trace, post_spike)
                rise = pre_trace * post_spike
            de = - self._etrace * self.dt / self.tau_syn + rise
            self._etrace += de

    def update(self, reward=None) -> None:
        """
        Update the synaptic weights based on the learning rule.
        """
        if self._learning_rule_type is not None:
            # For STDP Rule
            # trace_pre, trace_post = self._tile(self.pre_layer.get_trace(), self.post_layer.get_trace())
            # spk_pre, spk_post = self._tile(self.pre_layer.spike, self.post_layer.spike)
            # self.weights = self.learning_rule.update(self.weights, spk_pre, spk_post, trace_pre, trace_post, reward=reward)
            dw = self.learning_rule.update(self, reward=reward)
            self.weights += dw
        # elif self._learning_rule_type is None:
        #     return 
        # elif self._learning_rule_type == "ANN":
        #     dw = self.learning_rule.update(self, reward=reward)
        #     self.weights += dw
            # trace_pre, trace_post = self._tile(self.pre_layer.get_trace(), self.post_layer.get_trace())
            # spk_pre, spk_post = self._tile(self.pre_layer.spike, self.post_layer.spike)
            # self.weights = self.learning_rule.update(self.weights, spk_pre, spk_post, trace_pre, trace_post, reward=reward)
        else:
            # For Empty Rule
            # self.weights = self.learning_rule.update(self.weights)
            return
        
        # Clip the weights
        if self.clip_weights:
            self.weights = np.clip(self.weights, self.weight_min, self.weight_max)
        
        # Normalise the weights
        self._normalise_weights()

    def _normalise_weights(self):
        if self.normalise_weights:
            self.weights = safe_norm(self.weights, self.normalise_method)

    def __repr__(self):
        return f"SynapseLayer({self.weights.shape})"
    
    @property
    def eligibility_trace(self):
        """
        Return the eligibility trace if it is being used, otherwise return None.
        """
        if self._use_elig:
            return self._etrace
        else:
            return None
    

def safe_norm(array, method, params={}, eps=1e-10):
    if method == "sum":
        denom = np.sum(array, axis=0, keepdims=True)
    elif method == "L2":
        denom = np.linalg.norm(array, axis=0, keepdims=True)
    elif method == "P":
        p = params.get("p", 1)
        denom = np.linalg.norm(array, ord=p, axis=0, keepdims=True)
    else:
        raise ValueError(f"Normalisation method {method} not recognised.")
    
    # Avoid division by zero
    denom = np.where(denom == 0, eps, denom)
    array = array / denom
    return array

