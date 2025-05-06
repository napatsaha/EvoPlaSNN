import numpy as np
from .neurons import NeuronLayer
from .lrule import LearningRule, Empty_Rule, STDP_Rule


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
                 weight_init_params: dict = None, learning_rule: LearningRule = None):

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()

        # Initialize weights
        self.weight_init = weight_init
        self.weight_init_params = weight_init_params
        if self.weight_init_params is None:
            self.weight_init_params = {}
        if self.weight_init == 'uniform':
            low = self.weight_init_params.get('low', 0.0)
            high = self.weight_init_params.get('high', 1.0)
            self.weights = np.random.uniform(low, high, size=(self.pre_layer.size, self.post_layer.size))
        else:
            raise NotImplementedError("Other weight initialisation methods not implemented yet")

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
    
    def update(self):
        """
        Update the synaptic weights based on the learning rule.
        """
        if isinstance(self.learning_rule, STDP_Rule):
            trace_pre, trace_post = self._tile(self.pre_layer.get_trace(), self.post_layer.get_trace())
            spk_pre, spk_post = self._tile(self.pre_layer.spike, self.post_layer.spike)
            self.weights = self.learning_rule.update(self.weights, spk_pre, spk_post, trace_pre, trace_post)
        else:
            # For Empty Rule
            self.weights = self.learning_rule.update(self.weights)