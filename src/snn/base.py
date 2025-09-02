from typing import List, Protocol, Literal
import numpy as np
from abc import ABC, abstractmethod

from lrule.base import LearningRule

class SpikeGenerator(ABC):
    """
    Abstract base class for spike generators.
    """
    input_size: int
    rng: np.random.Generator
    count: int
    _current_class: int
    num_classes: int
    _full_length: int
    _pattern_length: int
    _finished: bool
    _static: bool

    def __init__(self, input_size: int, seed=None):
        self.input_size = input_size
        self.rng = np.random.default_rng(seed)
        self._static = True

    def reset(self):
        """
        Resets the generator to its initial state.
        This method should be overridden by subclasses if needed.
        """
        pass

    @abstractmethod
    def generate(self) -> np.ndarray:
        pass

    def generate_empty(self) -> np.ndarray:
        """
        Generates an empty spike train of zeros.
        """
        return np.zeros(self.input_size, dtype=np.int8)

    def return_signal(self) -> bool | int:
        """
        Depending on subclass, returns a signal that would help in updating synapses.
        """
        return False
    
    def get_label(self) -> None | int:
        """
        Returns the current class label.
        """
        return self._current_class

    def is_static(self) -> bool:
        """
        Determines whether the generator will always return the same spike train for each class,
        or if there is some stochasticity in how each pattern is presented.
        """
        return self._static
    
    def is_final(self) -> bool:
        """
        Returns whether or not this is the final timestep over the entire sample (including spacing).
        """
        return self.count >= self._full_length

    def update_classes(self, new_classes: List) -> None:
        """
        Allows spike generator to use a new set of classes of length num_classes.
        """
        pass

    def __len__(self):
        return self._full_length
    
    @property
    def length(self):
        """
        Returns the total number of time steps in a single pattern (including spacing).
        """
        return self._full_length

    @property
    def pattern_length(self):
        """
        Returns the duration of a single pattern (disregarding spacing in-between patterns).
        """
        return self._pattern_length
    
    @property
    def finished(self) -> bool:
        """
        Returns whether the current pattern is finished (including last time step in pattern).
        """
        return self._finished
    
    @property
    def active(self):
        """
        Returns whether the pattern is being generated or whether it is during a waiting period.
        """
        return (self.count <= self._pattern_length)

    @property
    def ready(self) -> bool:
        """
        Returns whether the generator is ready to allow weight updates or reward to be calculated.  
        """
        return self._finished and self.active
    
class NeuronLayerProtocol(Protocol):
    """
    Protocol class for NeuronLayer.
    Defines the public attributes and methods with their descriptions.
    """

    size: int
    dt: float
    membrane: np.ndarray
    spike: np.ndarray
    tssp: np.ndarray
    _trace: np.ndarray

    def __init__(self, size: int, *, tau_mem: float, tau_trace: float, dt: float, threshold: float, 
                 wta: bool, membrane_start: float, reset_mechanism: Literal["zero", "subtract"], 
                 trace_amp: float, trace_type: Literal["dx1", "dx2", "dx3"]) -> None:
        """
        Initialize the neuron layer with the given parameters.
        """
        pass

    def reset(self) -> None:
        """
        Reset the neuron layer state, including membrane potential, spike status, time since last spike, and trace.
        """
        pass

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """
        Update the neuron layer state based on the input current and time step.
        Returns the spike status as an array.
        """
        pass

    def get_trace(self) -> np.ndarray:
        """
        Return the trace of the neuron layer, which decays since the last spike.
        """
        pass

    def update_thresholds(self, delta_thr: np.ndarray):
        """
        Update the firing thresholds of the neurons by adding delta_thr.
        """
        pass



class SynapseLayerProtocol(Protocol):
    """
    Protocol abstract class for SynapseLayer.
    Defines the method and attribute names along with their descriptions.
    """

    pre_layer: NeuronLayerProtocol
    post_layer: NeuronLayerProtocol
    weights: np.ndarray
    learning_rule: LearningRule
    eligibility_trace: np.ndarray | None

    def __init__(self, pre_layer: NeuronLayerProtocol, post_layer: NeuronLayerProtocol, *,
                 learning_rule: LearningRule, eligibility_trace: bool, tau_syn: float, dt: float,
                 weight_init: str, weight_init_params: dict, weight_min: float, weight_max: float,
                 clip_weights: bool, normalise_weights: bool, normalise_method: Literal["sum", "L2", "P"],
                 normalise_params: dict) -> None:
        """
        Initialize the SynapseLayer with the given parameters.
        """
        pass

    def forward(self, spike_input: np.ndarray) -> np.ndarray:
        """
        Compute the output current to the next neuron layer given a spike current input from the previous neuron layer.
        """
        pass

    def update(self, reward: float | None) -> None:
        """
        Update the synaptic weights based on the learning rule.
        """
        pass

    def reset(self) -> None:
        """
        Reset the synaptic weights to their initial state.
        """
        pass

    def update_eligibility_trace(self) -> None:
        """
        Update the eligibility trace based on the pre and post neuron layer spikes.
        """
        pass



