
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure


def plot_spikes(ax: Axes, tf_spikes: np.ndarray, total_time: int, label: str = ""):

    ax.eventplot(tf_spikes, colors='gray', linelengths=0.5)
    ax.set_ylim(1.0, 1.5)
    ax.set_xlim(0, total_time)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center')

def plot_membrane_old(membrane_array, ax: Axes, threshold=None, title=None):
    ax.plot(membrane_array)
    if threshold is not None:
        ax.axhline(threshold, color='gray', linestyle='--')
    if title is not None:
        ax.set_title(title)

def plot_membranes(ax: Axes, mem: np.ndarray, *, threshold: float = None, tf_pre: int = None, tf_post: int = None,
                                title=None, xlabel=None, ylabel=None, **kwargs):
    T = len(mem)
    ymax = max(max(mem), 1.0)
    ymin = min(min(mem), 0.0)
    ymid = (ymax + ymin) / 2
    eps = (ymax - ymin) * 0.1

    ax.plot(mem, **kwargs)
    if tf_pre is not None:
        ax.vlines(x=tf_pre, ymin=ymin-eps, ymax=ymid - 2*eps, color='gray', alpha=0.5, linestyles='dotted')
    if tf_post is not None:
        ax.vlines(x=tf_post, ymin=ymid + 2*eps, ymax=ymax+eps, color='gray', alpha=0.7, linestyles='dotted')
    if threshold is not None:
        ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    ax.set_xlim(0, T)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(ymin - eps, ymax + eps)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

def plot_neuron(fig: Figure, gs: gridspec.GridSpec, mem: np.ndarray, *, tf_pre: int = None, tf_post: int = None, threshold: float = None, **kwargs):
    T = len(mem)
    
    ncols = 1# + int(tf_pre is not None) + int(tf_post is not None)
    nrows = 1
    height_ratios = [1.0]
    if tf_post is not None:
        height_ratios = [0.1] + height_ratios
        nrows += 1
    if tf_pre is not None:
        height_ratios = height_ratios + [0.1]
        nrows += 1

    gs0 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs, hspace=0.1, height_ratios=height_ratios)
    plot_idx = 0

    if tf_post is not None:
        ax = fig.add_subplot(gs0[plot_idx])
        plot_idx += 1
        plot_spikes(ax, tf_post, T, label="Post")

    ax = fig.add_subplot(gs0[plot_idx])
    plot_idx += 1
    plot_membranes(ax, mem, tf_pre=tf_pre, tf_post=tf_post, threshold=threshold, **kwargs)

    if tf_pre is not None:
        ax = fig.add_subplot(gs0[plot_idx])
        plot_idx += 1
        plot_spikes(ax, tf_pre, T, label="Pre")


class MatrixRecorder:
    def __init__(self, layer_shapes: List[tuple], total_timesteps: int, dtype=np.float32):
        self.layer_shapes = layer_shapes
        self.total_timesteps = total_timesteps
        self.dtype = dtype
        self.values = [np.zeros((layer_shape[0], layer_shape[1], total_timesteps), dtype=dtype) for layer_shape in layer_shapes]
    
    def record(self, layer_index: int, timestep: int, value: np.ndarray):
        assert value.shape == (self.layer_shapes[layer_index][0], self.layer_shapes[layer_index][1]), \
            f"Expected shape {(self.layer_shapes[layer_index][0], self.layer_shapes[layer_index][1])}, got {value.shape}"
        assert 0 <= layer_index < len(self.layer_shapes), \
            f"Layer index {layer_index} out of bounds for layer shapes {self.layer_shapes}"
        assert 0 <= timestep < self.total_timesteps, \
            f"Timestep {timestep} out of bounds for total timesteps {self.total_timesteps}"
        self.values[layer_index][:, :, timestep] = value



class LayerRecorder:
    def __init__(self, layer_sizes: List[int], total_timesteps: int, dtype=np.float32):
        self.layer_sizes = layer_sizes
        self.total_timesteps = total_timesteps
        self.dtype = dtype
        self.values = [np.zeros((layer_size, total_timesteps), dtype=dtype) for layer_size in layer_sizes]

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
                plot_membrane_old(layer[j, :], axs[j, i], threshold=thresholds[i], title=f"Layer {i}, Neuron {j}")

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
