
from typing import Literal
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
    def __init__(self, input_size: int, interval: float, *, spacing: int = None, 
                 signal_on_end: bool = False, starting_neuron: int = None,
                 ascending: bool = True, start_spike: bool = True, seed=None):
        super().__init__(input_size, seed)
        self.interval = max(1, int(interval))
        self.spacing = max(1, int(spacing)) if spacing is not None else interval
        self.signal_on_end = signal_on_end
        self.ascending = ascending
        self.start_spike = start_spike
        self.starting_neuron = max(0, min(input_size - 1, starting_neuron)) if starting_neuron is not None else 0 if ascending else input_size - 1
        self.direction = 1 if ascending else -1
        self.reset()
        # self.delay_count = self.interval if start_spike else 0
        # self.spike_count = 0
        # self.current_neuron = 0 if ascending else input_size - 1
        # self.finished = False

    def reset(self):
        self.delay_count = self.interval if self.start_spike else 0
        # if self.spacing > 0:
        # self.current_neuron = 0 if self.ascending else self.input_size - 1
        self.current_neuron = self.starting_neuron
        # else:
        #     self.current_neuron = - self.spacing if self.ascending else self.input_size + self.spacing - 1
        self.finished = False
        self.spike_count = 0

    def generate(self) -> np.ndarray:
        """
        Generates a spike pattern for the input layer.
        The pattern consists of spikes that occur at regular intervals.
        """
        self.delay_count += 1
        spikes =  self.generate_empty()

        # Intermission
        if self.finished:
            if self.delay_count >= self.spacing:
                # self.delay_count += 1
                self.reset()
                # spikes =  self.generate_empty()
            # else:
            # if self.delay_count >= (self.spacing - 1):
            #     self.reset()
        # Regular Spiking
        if not self.finished:
            if self.delay_count >= self.interval:
                # Reset delay count
                self.delay_count = 0

                # Create a spike pattern
                # spikes = self.generate_empty()
                spikes[self.current_neuron] = 1

                # Update the current neuron index
                self.current_neuron = (self.current_neuron + self.direction) % self.input_size
                self.spike_count += 1
                # Check if the pattern is finished
                if self.spike_count >= self.input_size:
                    self.finished = True
                    self.spike_count = 0

            # else:
            #     spikes = self.generate_empty()
            
        return spikes

    def return_signal(self) -> bool | int:
        """
        Returns a signal that indicates whether the pattern is finished.
        """
        if self.signal_on_end:
            return self.finished
        else:
            return True

    def __len__(self):
        return self.interval * (self.input_size - 1) + self.spacing




class BinaryClassGenerator(SpikeGenerator):
    """
    A SpikeGenerator that alternates between two classes of PatternSpikeGenerators with opposite directions.
    """
    def __init__(self, input_size, interval: int, *, spacing: int = None, start_spike: bool = True, 
                 signal_on_end: bool = False,
                 starting_class: Literal["ascending", "descending"] = "ascending", p: float = 0.5, seed=None):
        super().__init__(input_size, seed)
        self.p = p
        self.generators = [
            PatternSpikeGenerator(input_size, interval, spacing=spacing, ascending=True, start_spike=start_spike),
            PatternSpikeGenerator(input_size, interval, spacing=spacing, ascending=False, start_spike=start_spike)
        ]
        self.current_class = 0 if starting_class == "ascending" else 1
        self.signal_on_end = signal_on_end
        self.count = 0
        self.switch = False
        self._finished = False
        self._ready = False

    def _generate_spikes(self):
        self.count += 1
        gen = self.generators[self.current_class]
        spikes = gen.generate()
        finished = gen.finished
        self.switch = self.count >= len(gen)
        if self.switch:
            self._switch_class()
            self.generators[self.current_class].reset()
            self.switch = False
            self.count = 0
        return spikes, finished
    
    def _switch_class(self):
        r = np.random.rand()
        if r < self.p:
            self.current_class = 1 - self.current_class
        # self.current_class = int(r < self.p)
    
    def generate(self) -> np.ndarray:
        spikes, finished = self._generate_spikes()
        self._ready = finished and not self._finished
        self._finished = finished

        return spikes
        
    def return_signal(self):
        if self.signal_on_end:
            return self._ready
        else:
            return True

    def get_label(self) -> None | int:
        if not self._ready:
            return None
        else:
            return self.current_class



if __name__ == "__main__":
    gen = PatternSpikeGenerator(input_size=5, interval=2, spacing=5, ascending=False, start_spike=True)
    N = len(gen)
    result = []
    gen.reset()
    for i in range(N*2):
        spk = gen.generate()
        print(spk)
        print(f"t: {i:2}, Finished: {gen.finished}, Delay Count: {gen.delay_count}, Spike Count: {gen.spike_count}", end='\n\n')