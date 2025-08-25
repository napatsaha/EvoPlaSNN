"""
Simple Network of LIF neurons
"""

from typing import List, Literal
import numpy as np
from typing import Union
from .synapse import SynapseLayer
from lrule import LearningRule, Empty_Rule
from .neurons import NeuronLayer
from common.utils import solve_hidden


class SNN:
    neuron_layers: List[NeuronLayer]
    synapse_layers: List[SynapseLayer]

    def __init__(self, input_size: int, hidden_size: list[int] | int | None, output_size: int, *, dt: float = 1e-3, 
                 learning_rule: LearningRule = None, 
                 neuron_params: dict = None,
                 synapse_params: dict = None, 
                 winner_take_all: bool = False, soft_reset: bool = False,
                 sim_method: Literal["event-driven", "step-wise"] = "step-wise",
                #  tau_mem: float | List[float] = 5e-3, tau_trace: float | List[float] = 1e-1, threshold: float | List[float] = 1.0, 
                #  membrane_start: float = 0.0, reset_mechanism = "subtract"
                 ):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_size = solve_hidden(hidden_size)
        self.output_size = output_size
        self.layer_sizes = [input_size] + self.hidden_size + [output_size]
        self.layer_sizes_active = self.layer_sizes
        self.num_layers = len(self.layer_sizes_active)

        # Learning rule
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()

        # Simulation related parameters
        self.dt = dt
        self.sim_method = sim_method
        if self.sim_method not in ["event-driven", "step-wise"]:
            raise ValueError(f"Invalid simulation method: {self.sim_method}. Choose 'event-driven' or 'step-wise'.")

        # Store neuron and synapse parameters
        self.winner_take_all = winner_take_all
        self.neuron_params = [
            {k: v if not isinstance(v, list | tuple) else v[(self.num_layers + i) % len(v)] for k, v in neuron_params.items()} \
            for i in range(self.num_layers)
        ] if neuron_params is not None else [{}] * self.num_layers

        self.synapse_params = synapse_params if synapse_params is not None else {}
        self.use_etrace = self.synapse_params.get("eligibility_trace", False)

        # Create each neuron layers
        self.neuron_layers = []
        for i, layer_size in enumerate(self.layer_sizes_active):
            if self.winner_take_all and i > 0:
                self.neuron_params[i]["wta"] = True
            else:
                self.neuron_params[i]["wta"] = False
            layer = NeuronLayer(layer_size, dt=self.dt,
                                sim_method=self.sim_method, 
                                **self.neuron_params[i]
                                )
            self.neuron_layers.append(layer)

        # Create synapse layers
        self.synapse_layers = []
        for i in range(self.num_layers - 1):
            pre_layer = self.neuron_layers[i]
            post_layer = self.neuron_layers[i + 1]
            synapse = SynapseLayer(pre_layer, post_layer, learning_rule=self.learning_rule, 
                                   dt=self.dt, sim_method=self.sim_method,
                                   **self.synapse_params)
            self.synapse_layers.append(synapse)

        # Other parameters
        self._soft_reset = soft_reset
    
    def forward(self, spike_in:np.array) -> np.ndarray:
        curr = spike_in
        for i in range(self.num_layers - 1):
            spk = self.neuron_layers[i].forward(curr)
            curr = self.synapse_layers[i].forward(spk)
        spike_out = self.neuron_layers[-1].forward(curr)
        # Update eligibility traces if applicable
        if self.use_etrace:
            for synapse in self.synapse_layers:
                synapse.update_eligibility_trace()
        return spike_out

    def update_synapses(self, reward=None):
        for synapse in self.synapse_layers:
            synapse.update(reward)

    def reset(self):
        """
        Reset the state of the network, including all neuron and synapse layers.
        """
        for neuron_layer in self.neuron_layers:
            neuron_layer.reset()
        for synapse_layer in self.synapse_layers:
            synapse_layer.reset()

    def soft_reset(self):
        """
        Reset only neuron membrane potentials, spikes and  synapse eligibility traces.
        """
        for neuron_layer in self.neuron_layers:
            neuron_layer.reset()
        for synapse_layer in self.synapse_layers:
            synapse_layer.soft_reset()

    def use_soft_reset(self) -> bool:
        """
        Return whether the network uses soft reset or not
        """
        return self._soft_reset

    def get_exploration_rate(self, simplify: bool = False):
        values = []
        for neuron_layer in self.neuron_layers[1:]:
            values.append(neuron_layer.softmax_temp)
        if simplify:
            # Might not be the best way to do this
            if len(set(values)) == 1:
                return values[0]
        return np.asarray(values)

    def set_exploration_rate(self, value):
        for neuron_layer in self.neuron_layers[1:]:
            neuron_layer.softmax_temp = value

    def set_deterministic(self):
        for neuron_layer in self.neuron_layers[1:]:
            neuron_layer.spike_method = "deterministic"

    def set_stochastic(self):
        for neuron_layer in self.neuron_layers[1:]:
            neuron_layer.spike_method = "stochastic"

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
    
    @property
    def thresholds(self):
        return [layer.threshold for layer in self.neuron_layers]

    @property
    def eligibility_traces(self):
        return [synapse.eligibility_trace for synapse in self.synapse_layers]

    def __repr__(self):
        return f"SpikingNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

    def __str__(self):
        s = "SpikingNetwork(\n"
        for i in range(self.num_layers):
            s += f"  {self.neuron_layers[i]},\n"
            if i < self.num_layers - 1:
                s += f"  {self.synapse_layers[i]},\n"
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

