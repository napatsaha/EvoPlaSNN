
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_spikes(spike_in, spike_out=None):
    pass

def plot_membrane(membrane_array, ax: Axes, threshold=None, title=None):
    ax.plot(membrane_array)
    if threshold is not None:
        ax.axhline(threshold, color='gray', linestyle='--')
    if title is not None:
        ax.set_title(title)


class LayerRecorder:
    def __init__(self, layer_sizes: List[int], total_timesteps: int, dtype=np.float32):
        self.layer_sizes = layer_sizes
        self.total_timesteps = total_timesteps
        self.dtype = dtype
        self.values = [np.zeros((layer_size, total_timesteps), dtype=dtype) for layer_size in layer_sizes]

    def record(self, layer_index: int, timestep: int, membrane_potential: np.ndarray):
        assert membrane_potential.shape == (self.layer_sizes[layer_index],), \
            f"Expected shape {(self.layer_sizes[layer_index],)}, got {membrane_potential.shape}"
        assert 0 <= layer_index < len(self.layer_sizes), \
            f"Layer index {layer_index} out of bounds for layer sizes {self.layer_sizes}"
        assert 0 <= timestep < self.total_timesteps, \
            f"Timestep {timestep} out of bounds for total timesteps {self.total_timesteps}"
        self.values[layer_index][:, timestep] = membrane_potential

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
                plot_membrane(layer[j, :], axs[j, i], threshold=thresholds[i], title=f"Layer {i}, Neuron {j}")

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
