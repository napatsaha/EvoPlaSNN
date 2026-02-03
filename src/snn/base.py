from typing import List
import numpy as np
from abc import ABC, abstractmethod


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
    






