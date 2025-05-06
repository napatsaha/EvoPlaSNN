
import numpy as np

from abc import ABC, abstractmethod



class SpikeGenerator(ABC):
    """
    Abstract base class for spike generators.
    """

    def __init__(self, input_size: int, seed=None):
        self.input_size = input_size
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self) -> np.ndarray:
        pass


class RandomSpikeGenerator(SpikeGenerator):
    """
    Generates random spikes based on a given distribution.
    """
    # For now, only binomial distribution is supported
    _accepted_dist = ["binomial"]

    def __init__(self, input_size: int, dist: str = "binomial", seed=None, **kwargs):
        super().__init__(input_size, seed)
        if dist not in self._accepted_dist:
            raise ValueError(f"Distribution {dist} not supported")
        self.dist = dist
        # if dist == "binomial":
        #     self.function = self.rng.binomial
        self.dist_args = kwargs

    def generate(self):
        if self.dist == "binomial":
            return self.rng.binomial(1, **self.dist_args, size=self.input_size)
        else:
            raise NotImplementedError(f"Distribution {self.dist} not implemented")


class PatternSpikeGenerator(SpikeGenerator):
    """
    Generates a pattern of sequential spikes for each neuron in the input layer.
    Delays between spikes can be specified by the interval argument.
    """
    def __init__(self, input_size: int, interval: float, *, ascending: bool = True, start_spike: bool = True, seed=None):
        super().__init__(input_size, seed)
        self.interval = np.abs(interval)
        self.ascending = ascending
        self.delay_count = self.interval if start_spike else 0
        self.current_neuron = 0 if ascending else input_size - 1
        self.direction = 1 if ascending else -1

    def generate(self) -> np.ndarray:
        """
        Generates a spike pattern for the input layer.
        The pattern consists of spikes that occur at regular intervals.
        """
        if self.delay_count < (self.interval - 1):
            self.delay_count += 1
            return np.zeros(self.input_size, dtype=np.int8)

        else:
            # Reset delay count
            self.delay_count = 0

            # Create a spike pattern
            spikes = np.zeros(self.input_size, dtype=np.int8)
            spikes[self.current_neuron] = 1

            # Update the current neuron index
            self.current_neuron = (self.current_neuron + self.direction) % self.input_size

            return spikes
