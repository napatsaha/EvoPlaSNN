
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from .plot import _plot_membrane_old


def movmean(x: np.ndarray, n: int, mode: str = "valid") -> np.ndarray:   
    if x.ndim == 1:
        return np.convolve(x, np.ones(n)/n, mode=mode)
    elif x.ndim == 2:
        return np.array([np.convolve(x[i], np.ones(n)/n, mode=mode) for i in range(x.shape[0])])
    else:
        raise ValueError(f"movmean only supports 1D and 2D arrays, got {x.ndim}D array instead.")
    


class MatrixRecorder:
    def __init__(self, layer_shapes: List[tuple], total_timesteps: int = 0, dtype=np.float32):
        self.layer_shapes: List[tuple] = layer_shapes
        self.total_timesteps = total_timesteps
        self.dtype = dtype
        self.values = [np.zeros((layer_shape[0], layer_shape[1], total_timesteps), dtype=dtype) for layer_shape in layer_shapes]
    
    def setup(self, num_steps: int):
        self.total_timesteps += num_steps
        # Concatenate the existing values with new zeros
        for i, layer_shape in enumerate(self.layer_shapes):
            z = np.zeros((layer_shape[0], layer_shape[1], num_steps), dtype=self.dtype)
            self.values[i] = np.concatenate((self.values[i], z), axis=2)

    def reset(self):
        self.total_timesteps = 0
        for i, layer_shape in enumerate(self.layer_shapes):
            self.values[i] = np.zeros((layer_shape[0], layer_shape[1], self.total_timesteps), dtype=self.dtype)

    def record(self, layer_index: int, timestep: int, value: np.ndarray):
        if self.total_timesteps == 0:
            raise ValueError("Recorder has not been initialized with total_timesteps.")
        assert value.shape == (self.layer_shapes[layer_index][0], self.layer_shapes[layer_index][1]), \
            f"Expected shape {(self.layer_shapes[layer_index][0], self.layer_shapes[layer_index][1])}, got {value.shape}"
        assert 0 <= layer_index < len(self.layer_shapes), \
            f"Layer index {layer_index} out of bounds for layer shapes {self.layer_shapes}"
        assert 0 <= timestep < self.total_timesteps, \
            f"Timestep {timestep} out of bounds for total timesteps {self.total_timesteps}"
        self.values[layer_index][:, :, timestep] = value



class LayerRecorder:
    def __init__(self, layer_sizes: List[int], total_timesteps: int = 0, dtype=np.float32):
        self.layer_sizes = layer_sizes
        self.total_timesteps = total_timesteps
        self.dtype = dtype
        self.values = [np.zeros((layer_size, total_timesteps), dtype=dtype) for layer_size in layer_sizes]
    
    def setup(self, num_steps: int):
        self.total_timesteps += num_steps
        # Concatenate the existing values with new zeros
        for i, layer_size in enumerate(self.layer_sizes):
            z = np.zeros((layer_size, num_steps), dtype=self.dtype)
            self.values[i] = np.concatenate((self.values[i], z), axis=1)

    def reset(self):
        self.total_timesteps = 0
        for i, layer_size in enumerate(self.layer_sizes):
            self.values[i] = np.zeros((layer_size, self.total_timesteps), dtype=self.dtype)

    def record(self, layer_index: int, timestep: int, value: np.ndarray):
        assert value.shape == (self.layer_sizes[layer_index],), \
            f"Expected shape {(self.layer_sizes[layer_index],)}, got {value.shape}"
        assert 0 <= layer_index < len(self.layer_sizes), \
            f"Layer index {layer_index} out of bounds for layer sizes {self.layer_sizes}"
        assert 0 <= timestep < self.total_timesteps, \
            f"Timestep {timestep} out of bounds for total timesteps {self.total_timesteps}"
        self.values[layer_index][:, timestep] = value

    def plot(self, thresholds: int | List[int] = None, figtitle: str = None, col_width: float = 5.0, row_height: float = 2.5,
             savepath: str | Path = None, show: bool = True):
        # Setting up the figure
        ncols = len(self.layer_sizes)
        nrows = max(self.layer_sizes)
        fig, axs = plt.subplots(nrows, ncols, figsize=(col_width*ncols, row_height*nrows), sharex=True, sharey=True)

        # Handling thresholds
        if thresholds is not None:
            if isinstance(thresholds, list):
                assert len(thresholds) == len(self.layer_sizes), \
                    f"Length of thresholds {len(thresholds)} does not match number of layers {len(self.layer_sizes)}"
            else:
                thresholds = [thresholds for _ in range(ncols)]
        else:
            thresholds = [None for _ in range(ncols)]

        # Plotting
        for i, layer in enumerate(self.values):
            for j in range(layer.shape[0]):
                _plot_membrane_old(layer[j, :], axs[j, i], threshold=thresholds[i], title=f"Layer {i}, Neuron {j}")

        # Formatting
        if figtitle is not None:
            fig.suptitle(figtitle, fontsize=16)
        fig.tight_layout()

        # Display and saving
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close(fig)
