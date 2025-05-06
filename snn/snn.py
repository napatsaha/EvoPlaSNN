"""
Simple Network of LIF neurons
"""

from typing import List, Literal
import numpy as np
from typing import Union
from .synapse import SynapseLayer
from .lrule import LearningRule, Empty_Rule
from .neurons import NeuronLayer


class SNN:
    neuron_layers: List[NeuronLayer]
    synapse_layers: List[SynapseLayer]

    def __init__(self, input_size: int, hidden_size: list[int] | int | None, output_size: int, dt, *, 
                 learning_rule: LearningRule = None, synapse_params: dict = None, neuron_params: dict = None,
                #  tau_mem: float | List[float] = 5e-3, tau_trace: float | List[float] = 1e-1, threshold: float | List[float] = 1.0, 
                #  membrane_start: float = 0.0, reset_mechanism = "subtract"
                 ):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        if hidden_size is None or len(hidden_size) == 0 or hidden_size == 0:
            self.hidden_size = []
        elif isinstance(hidden_size, int):
            self.hidden_size = [hidden_size]
        elif isinstance(hidden_size, list):
            self.hidden_size = hidden_size
        else:
            raise TypeError("hidden_size must be an int or a list of ints.")
        self.output_size = output_size
        self.layer_sizes = [input_size] + self.hidden_size + [output_size]
        self.layer_sizes_active = self.layer_sizes
        self.num_layers = len(self.layer_sizes_active)

        # Learning rule
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()

        # Time related parameters
        self.dt = dt

        # Store neuron and synapse parameters
        # self.tau_mem = tau_mem
        # self.tau_trace = tau_trace
        # self.threshold = threshold
        # self.membrane_start = membrane_start
        # self.reset_mechanism = reset_mechanism
        self.neuron_params = [
            {k: v if not isinstance(v, list | tuple) else v[(self.num_layers + i) % len(v)] for k, v in neuron_params.items()} \
            for i in range(self.num_layers)
        ]

        self.synapse_params = synapse_params if synapse_params is not None else {}

        # Create each neuron layers
        self.neuron_layers = []
        for i, layer_size in enumerate(self.layer_sizes_active):
            layer = NeuronLayer(layer_size, dt=self.dt, **self.neuron_params[i]
                                # tau_mem=tau_mem if isinstance(tau_mem, float) else tau_mem[i],
                                # tau_trace=tau_trace if isinstance(tau_trace, float) else tau_trace[i],
                                # threshold=threshold if isinstance(threshold, float) else threshold[i],
                                # membrane_start=membrane_start, reset_mechanism=reset_mechanism
                                )
            self.neuron_layers.append(layer)

        # Create synapse layers
        self.synapse_layers = []
        for i in range(self.num_layers - 1):
            pre_layer = self.neuron_layers[i]
            post_layer = self.neuron_layers[i + 1]
            synapse = SynapseLayer(pre_layer, post_layer, learning_rule=self.learning_rule, **self.synapse_params)
            self.synapse_layers.append(synapse)
    
    def forward(self, spike_in):
        curr = spike_in
        for i in range(self.num_layers - 1):
            spk = self.neuron_layers[i].forward(curr)
            curr = self.synapse_layers[i].forward(spk)
        spike_out = self.neuron_layers[-1].forward(curr)
        return spike_out

    def update_synapses(self):
        for synapse in self.synapse_layers:
            synapse.update()

    @property
    def membranes(self):
        return [layer.membrane for layer in self.neuron_layers]
    
    @property
    def spikes(self):
        return [layer.spike for layer in self.neuron_layers]
    
    @property
    def traces(self):
        return [layer.get_trace() for layer in self.neuron_layers]
    
    @property
    def weights(self):
        return [synapse.weights for synapse in self.synapse_layers]

    def __repr__(self):
        return f"SpikingNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

    def __str__(self):
        s = "SpikingNetwork(\n"
        for w in self.weights:
            s += f"  {w.shape}\n"
        s += ")\n"
        return s
   


if __name__ == "__main__":
    # Example usage
    input_size = 10
    hidden_size = [5]
    output_size = 2
    tau = 1e-3
    dt = 5e-3
    threshold = 1.0

