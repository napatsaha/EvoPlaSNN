from typing import Literal
import numpy as np
from .neurons import NeuronLayer
from lrule import ANN_Rule, LearningRule, Empty_Rule, STDP_Rule


class SynapseLayer:
    """
    Purpose: To represent a collection of synapse that connects one layer of neurons to another.
    (Assuming a Sequential Linear Layer network with all-to-all connections)

    Functionality:
    `forward()`: computes the output current to next neuron layer given a spike current input 
    from previous neuron layer -- i.e. weighted spike current
    `update()`: Update the entire synaptic weight efficacies based on a particular LearningRule
    """
    post_layer: NeuronLayer
    pre_layer: NeuronLayer
    weights: np.ndarray
    learning_rule: LearningRule
    
    def __init__(self, pre_layer: NeuronLayer, post_layer: NeuronLayer, weight_init: str = 'uniform',
                 weight_init_params: dict = None, learning_rule: LearningRule = None,
                 weight_min: float = 0.0, weight_max: float = 1.0,
                 clip_weights: bool = True, 
                 normalise_weights: bool = False, normalise_method: Literal["sum", "L2", "P"] = "sum", normalise_params: dict = None):

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()
        self._learning_rule_type = self._get_lrule_type()

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
    
    def forward(self, spike_input: np.ndarray) -> np.ndarray:
        """
        Compute the output current to the next neuron layer given a spike current input from the previous neuron layer.
        """
        assert spike_input.shape[0] == self.pre_layer.size
        assert spike_input.ndim == 1

        # Compute the output current
        output_current = np.dot(spike_input, self.weights)
        return output_current
    
    def update(self, reward=None):
        """
        Update the synaptic weights based on the learning rule.
        """
        if self._learning_rule_type == "STDP":
            # For STDP Rule
            trace_pre, trace_post = self._tile(self.pre_layer.get_trace(), self.post_layer.get_trace())
            spk_pre, spk_post = self._tile(self.pre_layer.spike, self.post_layer.spike)
            self.weights = self.learning_rule.update(self.weights, spk_pre, spk_post, trace_pre, trace_post, reward=reward)
        elif self._learning_rule_type is None:
            # For Empty Rule
            return 
            # self.weights = self.learning_rule.update(self.weights)
        else:
            return
        
        # Clip the weights
        if self.clip_weights:
            self.weights = np.clip(self.weights, self.weight_min, self.weight_max)
        
        # Normalise the weights
        if self.normalise_weights:
            if self.normalise_method == "sum":
                self.weights = self.weights / np.sum(self.weights, axis=0, keepdims=True)
            elif self.normalise_method == "L2":
                self.weights = self.weights / np.linalg.norm(self.weights, axis=0, keepdims=True)
            elif self.normalise_method == "P":
                p = self.normalise_params.get("p", 1)
                self.weights = self.weights / np.linalg.norm(self.weights, ord=p, axis=0, keepdims=True)
            else:
                raise ValueError(f"Normalisation method {self.normalise_method} not recognised.")
            

    def __repr__(self):
        return f"SynapseLayer({self.weights.shape})"