"""
2026-08-03

Spike Encoding / Decoding Schemes. 

Based on system in:
Schuman, C., Rizzo, C., McDonald-Carmack, J., Skuda, N., & Plank, J. (2022). 
Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems. 
Proceedings of the International Conference on Neuromorphic Systems 2022, 1–9. 
https://doi.org/10.1145/3546790.3546792

"""

from common.base import SpikeCoder
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Literal


class BaseSpikeCoder(SpikeCoder, ABC):
    def __init__(self, input_channels, output_channels, *,
                 encoding_method: Literal["single", "multi", "rate", "temporal"] = 'single',
                 n_neurons_in=1, 
                 n_neurons_out=None,
                 upper_bounds, lower_bounds,
                 window_size):
        super().__init__()
        self._input_channels = input_channels
        self._n_neurons_in = min(1, n_neurons_in)
        self._input_neurons = self._input_channels * self._n_neurons_in

        self._output_channels = output_channels
        self._n_neurons_out = min(1, n_neurons_out)
        self._output_neurons = self._output_channels * self._n_neurons_out


    def encode(self, inp: int | np.ndarray):
        self._validate_input(inp)
        pass

    def decode(self, spikes):
        pass

    def reset(self):
        pass

    def _validate_input(self, inp):
        pass

    @property
    def ready(self):
        return super().ready

    @property
    def input_size(self):
        return self._input_neurons

    @property
    def output_size(self):
        return self._output_neurons


class BaseSpikeEncoder(ABC):
    def __init__(self, n_channels, n_neurons, window_size, lower_bounds, upper_bounds):
        self.n_channels = n_channels
        self.n_neurons = n_neurons
        self.window_size = window_size
        assert len(lower_bounds) == self.n_channels
        self.lower_bounds = lower_bounds
        assert len(upper_bounds) == self.n_channels
        self.upper_bounds = upper_bounds

        self._empty_array = np.zeros((self.n_channels, self.n_neurons, self.window_size), dtype=np.int8)
        self._bins = [np.linspace(lw, hg, self.n_neurons) for lw, hg in zip(self.lower_bounds, self.upper_bounds)]

    def _get_neuron_index(self, inp):
        """
        Get index of corresponding neuron for each channel based on defined bins

        Args:
            inp (_type_): _description_
        """
        return [np.digitize(xi, bin_i) for xi, bin_i in zip(inp, self._bins)]

    @abstractmethod
    def generate_spikes(self, inp):
        pass


class BaseSpikeDecoder(ABC):
    def __init__(self, n_channels, n_neurons, window_size, lower_bounds, upper_bounds):
        self.n_channels = n_channels
        self.n_neurons = n_neurons
        self.window_size = window_size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    @abstractmethod
    def decode_spikes(self, inp):
        pass


class SingleSpikeEncoder(BaseSpikeEncoder):
    def __init__(self, n_channels, n_neurons, window_size, lower_bounds, upper_bounds):
        super().__init__(n_channels, n_neurons, window_size, lower_bounds, upper_bounds)

    def generate_spikes(self, inp):
        neuron_idx = self._get_neuron_index(inp)
        out_array = self._empty_array.copy()
        out_array[np.arange(self.n_channels), neuron_idx, 0] = 1
        return out_array
