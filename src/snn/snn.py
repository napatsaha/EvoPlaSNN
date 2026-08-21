"""
Simple Network of LIF neurons
"""

from typing import List, Literal, Sequence
from common.base import LearningRule
import numpy as np
from typing import Union
from .synapse import SynapseLayer
from lrule import Empty_Rule
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
                 update_weights_on_etrace: bool | Literal["pre", "post", "stdp", "custom"] = False,
                 update_lrate: float = 1.0,
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
        self._learning_rule = learning_rule if learning_rule is not None else Empty_Rule()

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
        self.use_etrace_pre = self.synapse_params.get("eligibility_trace", False) or self.synapse_params.get("eligibility_pre", False)
        self.use_etrace_post = self.synapse_params.get("eligibility_post", False)
        self.use_etrace_stdp = self.synapse_params.get("eligibility_stdp", False)
        self.use_etrace_custom = self.synapse_params.get("eligibility_custom", False)
        self.use_etrace = self.use_etrace_pre or self.use_etrace_post or self.use_etrace_stdp

        self.update_weights_on_etrace = bool(update_weights_on_etrace)
        self._etrace_to_update = str(update_weights_on_etrace) if self.update_weights_on_etrace else None
        self._update_lrate = float(update_lrate)

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
            synapse = SynapseLayer(pre_layer, post_layer, learning_rule=self._learning_rule, 
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

    def apply_learning_rule(self, reward=None):
        for synapse in self.synapse_layers:
            synapse.apply_learning_rule(reward)

    def apply_weight_updates_from_etrace(self, signal: float = None, lrate: float = 1.0):
        for synapse in self.synapse_layers:
            synapse.update_weights_from_etrace(signal, self._etrace_to_update, lrate=self._update_lrate)

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
            neuron_layer.soft_reset()
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

    def set_exploration_rate(self, value: float = None):
        if value is not None:
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
    def eligibility_traces_pre(self):
        return [synapse.eligibility_pre for synapse in self.synapse_layers]
    
    @property
    def eligibility_traces_post(self):
        return [synapse.eligibility_post for synapse in self.synapse_layers]
    
    @property
    def eligibility_traces_stdp(self):
        return [synapse.eligibility_stdp for synapse in self.synapse_layers]

    @property
    def eligibility_traces_custom(self):
        return [synapse.eligibility_custom for synapse in self.synapse_layers]
    
    @property
    def learning_rule(self):
        return self._learning_rule
    
    @learning_rule.setter
    def learning_rule(self, rule: LearningRule):
        self._learning_rule = rule
        for synapse in self.synapse_layers:
            synapse.learning_rule = rule

    @property
    def tau_trace(self) -> List[float]:
        return [layer.tau_trace for layer in self.neuron_layers]
    @tau_trace.setter
    def tau_trace(self, value: int | float | Sequence | np.ndarray):
        if isinstance(value, (Sequence, np.ndarray)):
            assert len(value) == len(self.neuron_layers)
            for v, neuron_layer in zip(value, self.neuron_layers):
                neuron_layer.tau_trace = v
        else:
            for neuron_layer in self.neuron_layers:
                neuron_layer.tau_trace = value

    @property
    def tau_mem(self) -> List[float]:
        return [layer.tau_mem for layer in self.neuron_layers]
    @tau_mem.setter
    def tau_mem(self, value: int | float | Sequence):
        if isinstance(value, (Sequence, np.ndarray)):
            assert len(value) == len(self.neuron_layers)
            for v, neuron_layer in zip(value, self.neuron_layers):
                neuron_layer.tau_mem = v
        else:
            for neuron_layer in self.neuron_layers:
                neuron_layer.tau_mem = value

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

