from pathlib import Path
import csv, pickle
from typing import List, Literal, Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np
import os

import pandas as pd
import seaborn as sns

# from .simulate import SNNSimulator
from snn.utils import get_spike_times
from common import base
from common.utils import get_boundaries_for_lrule_inputs, LRULE_INPUT_BOUNDS

### Plotting functions that require Simulator ###


def plot_spikes(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, *, x_scale: float = 0.2, y_scale: float = 0.5,
                y_eps: float = 0.5, x_eps: float | int = 1, spk_eps: float = 0.25, 
                title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                linewidth=2, x_min = None, x_max = None, x_range: int = 100,
                figsize: tuple = None, dpi: int = 100,
                savepath: str | Path = None, show: bool = True, **kwargs):
    """
    Plot spike trains with time on x-axis and neuron index on y-axis.
    """
    # Flexible input handling
    if simulator is not None:
        assert simulator.record_spikes, "Spike recording is not enabled."
    num_steps = simulator.num_steps if simulator is not None else min([val.shape[1] for val in values])
    values = simulator.spike_recorder.values if simulator is not None else values
    if values is None:
        raise ValueError("Either a simulator (with spike recording enabled) or a list of recorded spikes [layer_size, num_steps] * num_layers, must be provided.")
    num_layers = simulator.network.num_layers if simulator is not None else len(values)
    layer_sizes = simulator.network.layer_sizes if simulator is not None else [val.shape[0] for val in values]
    dt = f"{simulator.dt} s" if simulator is not None else "1 unit"

    if x_eps < 1 and x_eps > 0:
        x_eps = x_eps * num_steps
    if x_max is None:
        x_max = num_steps
    if x_min is None:
        x_min = max(0, x_max - x_range)
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]        

    if figsize is None:
        figsize = ((x_max - x_min) * x_scale, sum(layer_sizes) * y_scale)
    fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                            figsize=figsize, dpi=dpi, layout="constrained")
    for i, layer_spikes in enumerate(reversed(values)):
        ax = axs[i]
        n_neurons = layer_spikes.shape[0]
        n_id, t_spk = np.where(layer_spikes)
        # ax.scatter(t_spk, n_id, marker="|")
        # if cmap is None:
        #     cm = "black"
        # else:
        #     cm = mpl.colormaps[cmap](n_id / (n_neurons - 1))
        if cmap is None:
            c = cm
        else:
            c = np.interp(n_id / (n_neurons - 1), (0, 1), cmap_range)
            c = cm(c)
        ax.vlines(t_spk, n_id - spk_eps, n_id + spk_eps, color=c, linewidth=linewidth, **kwargs)
        # ax.vlines(spike_times, i, i+1, color='black', alpha=0.5)
        ax.set_ylim(0 - y_eps, n_neurons - 1 + y_eps)
        ax.set_xlim(x_min - x_eps, x_max + x_eps)
        ax.set_yticks(np.arange(n_neurons))
        ax.set_yticklabels(np.arange(n_neurons))
        ax.set_ylabel(f"Layer {num_layers - i}", rotation=90, ha="center")
        # X Grid
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)
    ax.set_xlabel(f"Time ({dt})")
    # ax.set_ylabel("Neuron Index")
    fig.suptitle(title if title is not None else "Spike Trains", fontsize=20)
    fig.supylabel("Neuron Index")
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_traces(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, *, 
                x_scale: float = 0.2, y_scale: float = 0.8,
                y_eps: float = 0.1, x_eps: int | float = 1, trace_scale: float = 0.8, 
                x_min = None, x_max = None, x_range: int = 100,
                drawstyle: str = 'steps-post',
                title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                figsize: tuple = None, dpi: int = 100,
                savepath: str | Path = None, show: bool = True, **kwargs):
    """
    Plot traces
    """
    if simulator is not None:
        assert simulator.record_traces, "Trace recording is not enabled."
    if values is None and simulator is None:
        raise RuntimeError("Either a simulator (with trace recording enabled) or a list of recorded traces [layer_size, num_steps] * num_layers, must be provided.")

    # Extract necessary info
    num_steps = simulator.num_steps if simulator is not None else min([val.shape[1] for val in values])
    values = simulator.trace_recorder.values if simulator is not None else values
    num_layers = simulator.network.num_layers if simulator is not None else len(values)
    layer_sizes = simulator.network.layer_sizes if simulator is not None else [val.shape[0] for val in values]
    dt = f"{simulator.dt} s" if simulator is not None else "1 unit"

    if x_eps < 1 and x_eps > 0:
        x_eps = x_eps * num_steps
    if x_max is None:
        x_max = num_steps
    if x_min is None:
        x_min = max(0, x_max - x_range)
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]


    if figsize is None:
        figsize = ((x_max - x_min) * x_scale, sum(layer_sizes) * y_scale)
    fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                            figsize=figsize, dpi=dpi, layout="constrained")
    for i, layer_traces in enumerate(reversed(values)):
        ax = axs[i]
        n_neurons = layer_traces.shape[0]

        for j in range(n_neurons):
            if cmap is None:
                c = cm
            else:
                c = np.interp(j / (n_neurons - 1), (0, 1), cmap_range)
                c = cm(c)
            y = layer_traces[j, :]
            y = np.interp(y, (0, y.max()), (j, j + trace_scale))
            ax.plot(y, color=c, alpha=1.0, drawstyle=drawstyle, **kwargs)

        ax.set_ylim(0 - y_eps, n_neurons + y_eps)
        ax.set_xlim(x_min - x_eps, x_max + x_eps)
        ax.set_yticks(np.arange(n_neurons))
        ax.set_yticklabels(np.arange(n_neurons))
        ax.set_ylabel(f"Layer {i}", rotation=90, ha="center")
        # X Grid
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)

    ax.set_xlabel(f"Time ({dt} s)")
    fig.suptitle(title if title is not None else "Neuron Traces", fontsize=20)
    fig.supylabel("Neuron Index")
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_membranes(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, spike_values: List[np.ndarray] = None, 
                   thresholds: List[float | np.ndarray] = None, *, neuron_layer: int = None,
                   x_scale: float = 0.2, y_scale: float = 3.0, title: str = None, plot_inputs: bool = False, 
                   color: str = "blue", cmap: str = None, cmap_range: tuple = (0, 1), layout: str = "constrained",
                   x_min = None, x_max = None, x_range: int = 100,
                   figsize: tuple = None, dpi: int = 100,
                   savepath: str | Path = None, show: bool = True):
    """
    Plot membrane potentials of all neurons in the network.
    """
    if values is None and simulator is None:
        raise ValueError("Either a simulator (with membrane recording enabled) or a list of recorded values of shape: [layer_size, num_steps] * num_layers, must be provided.")    # Flexible input handling
    if simulator is not None:
        # assert simulator.record_spikes, "Spike recording is not enabled."
        assert simulator.record_membrane, "Membrane recording is not enabled."
        num_steps = simulator.num_steps
        values = simulator.mem_recorder.values
        spike_values = simulator.spike_recorder.values if simulator.record_spikes else None
        spike_times = get_spike_times(spike_values) if simulator.record_spikes else None
        thresholds = simulator.network.thresholds if not simulator.record_thresholds else simulator.threshold_recorder.values
        num_layers = simulator.network.num_layers 
        layer_sizes = simulator.network.layer_sizes 
        dt = f"{simulator.dt} s" 
    else:
        num_steps = min([val.shape[1] for val in values])
        values = values
        spike_values = spike_values
        spike_times = get_spike_times(spike_values) if spike_values is not None else None
        thresholds = thresholds
        num_layers = len(values)
        layer_sizes =  [val.shape[0] for val in values]
        dt =  "1 unit"

    if x_max is None:
        x_max = num_steps
    if x_min is None:
        x_min = max(0, x_max - x_range)
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]

    # Decide which layers to plot
    if neuron_layer is None:
        layers_to_plot = []
        for n in range(num_layers):
            if not plot_inputs and n==0:
                continue
            layers_to_plot.append(n)
    else:
        assert 0 <= neuron_layer < num_layers, f"Invalid 'neuron_layer={neuron_layer}' index. Must be between [0, {num_layers})"
        layers_to_plot = [neuron_layer]

    # Apply the designated layers_to_plot on relevant values
    if thresholds is not None:
        thresholds = [thresholds[i] for i in layers_to_plot]
    if spike_times is not None:
        spike_times = [spike_times[i] for i in layers_to_plot]
    values = [values[i] for i in layers_to_plot]
    layer_sizes = [layer_sizes[i] for i in layers_to_plot]
    # if not plot_inputs:
    #     thresholds = thresholds[1:] if thresholds is not None else None
    #     spike_times = spike_times[1:] if spike_times is not None else None
    #     values = values[1:]
    #     layer_sizes = layer_sizes[1:]
    # Construct figure based on sizes of layers to plot
    nrows = max(layer_sizes)
    ncols = len(layer_sizes)
    if figsize is None:
        figsize=(x_scale*ncols*(x_max - x_min), y_scale*nrows)
    fig, axs = plt.subplots(figsize=figsize, dpi=dpi, layout=layout)
    axs.remove()
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(ncols):
        layer_mem = values[i]
        n_neurons = layer_mem.shape[0]
        for j in range(n_neurons):
            if thresholds is not None:
                if thresholds[i].ndim == 1:
                    thr = float(thresholds[i][j])
                else:
                    thr = thresholds[i][j, :]
            else:
                thr = None
            if cmap is None:
                c = cm
            else:
                c = np.interp(j / (n_neurons - 1), (0, 1), cmap_range)
                c = cm(c)
            _plot_neuron(fig, gs[j, i], mem=layer_mem[j, :],
                        threshold=thr, 
                        tf_post=spike_times[i][j] if spike_times is not None else None, 
                        color=c, y_label=f"Neuron {j}",
                        x_min=x_min, x_max=x_max)
            
    # Labelling
    fig.text(s=f"Time ({dt})", fontsize=12, x=0.5, y=-0.01)
    fig.text(s="Membrane Potential", fontsize=12, ha='center', x=-0.01, y=0.5, rotation=90)
    fig.text(s=title if title is not None else "Membrane Potentials", fontsize=20, x=0.5, y=1.05)
    
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def _plot_neuron(fig: Figure, gs: gridspec.GridSpec, mem: np.ndarray, *, 
                 tf_pre: int = None, tf_post: int = None, threshold: float = None, y_label: str = None,
                 x_min: int = None, x_max: int = None, **kwargs):
    T = len(mem)
    x_min = 0 if x_min is None else x_min
    x_max = T if x_max is None else x_max
    
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
        _plot_spikes_single(ax, tf_post, label="Post", x_min=x_min, x_max=x_max)

    ax = fig.add_subplot(gs0[plot_idx])
    plot_idx += 1
    _plot_membrane_single(ax, mem, tf_pre=tf_pre, tf_post=tf_post, threshold=threshold, x_min=x_min, x_max=x_max, ylabel=y_label, **kwargs)

    if tf_pre is not None:
        ax = fig.add_subplot(gs0[plot_idx])
        plot_idx += 1
        _plot_spikes_single(ax, tf_pre, label="Pre", x_min=x_min, x_max=x_max)


def _plot_membrane_single(ax: Axes, mem: np.ndarray, *, threshold: np.ndarray | float = None, tf_pre: int = None, tf_post: int = None,
                                title=None, xlabel=None, ylabel=None, x_min: int = None, x_max: int = None, **kwargs):
    T = len(mem)
    x_min = 0 if x_min is None else x_min
    x_max = T if x_max is None else x_max
    ymax = max(max(mem), 1.0)
    ymin = min(min(mem), 0.0)
    ymid = (ymax + ymin) / 2
    eps = (ymax - ymin) * 0.1

    ax.plot(mem, **kwargs)
    if threshold is not None:
        if isinstance(threshold, float):
            ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
            thr_max, thr_min = threshold, threshold
        else:
            ax.plot(threshold, drawstyle='steps-post', color='black', linestyle='--', alpha=0.5)
            thr_max, thr_min = max(threshold), min(threshold)
        ymax = max(ymax, thr_max)
        ymin = min(ymin, thr_min)
    if tf_pre is not None:
        ax.vlines(x=tf_pre, ymin=ymin-eps, ymax=ymid - 2*eps, color='gray', alpha=0.5, linestyles='dotted')
    if tf_post is not None:
        ax.vlines(x=tf_post, ymin=ymid + 2*eps, ymax=ymax+eps, color='gray', alpha=0.7, linestyles='dotted')
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(ymin - eps, ymax + eps)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)


def _plot_spikes_single(ax: Axes, tf_spikes: np.ndarray, x_max: int, label: str = "", x_min: int = 0):

    ax.eventplot(tf_spikes, colors='gray', linelengths=0.5)
    ax.set_ylim(1.0, 1.5)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center')


# def plot_weights(simulator: 'SNNSimulator', div: int = 5, col_width: float = 6.0, row_height: float = 8.0,
#                 title: str = None, cmap: str = "gray",
#                 savepath: str | Path = None, show: bool = True):
#     assert simulator.record_weights, "Weight recording is not enabled."
#     num_layers = len(simulator.network.synapse_layers)

#     ts = np.linspace(0, simulator.num_steps-1, div+1).astype(int)

#     fig, axs = plt.subplots(num_layers, div+1, figsize=(col_width*div, row_height), squeeze=False)
#     fs = np.prod(fig.get_size_inches())/16
#     # mpl.rcParams.update({"font.size": np.prod(fig.get_size_inches())/16})

#     cmap = mpl.colormaps[cmap].reversed()

#     for l in range(num_layers):
#         for i, t in enumerate(ts):
#             im = simulator.weight_recorder.values[l][:, :, t]
#             ax = axs[l, i]
#             ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
#             ax.set_title(f"t={t}", fontsize=fs*0.8)
#             ax.set(xticks=[], yticks=[])

#     fig.supxlabel("Post-synaptic Neuron", y=0.2, fontsize=fs)
#     fig.supylabel("Pre-synaptic Neuron", x=0.1, fontsize=fs)
#     fig.suptitle(title if title is not None else "Synaptic Weights", fontsize=1.2*fs, y=0.995)
#     fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap), ax=axs,  
#                     orientation="horizontal", fraction=0.05, aspect=100, label="Weight")
#     fig.subplots_adjust(wspace=0.0, left=0.1, bottom=0.2)
#     if savepath is not None:
#         plt.savefig(savepath)
#     if show:
#         plt.show()
#     # plt.close(fig)

def plot_weights(simulator: "SNNSimulator" = None, values: List[np.ndarray] = None, *, env = None,
                 x_scale: float = 1.0, y_scale: float = 1.0, cmap: str = "viridis", bounded_weights: bool = True,
                 color_scale: str = "linear", figsize=None, dpi: int = 100,
                 savepath: str | Path = None, show: bool = True):
    """
    Plot the current synaptic weights of all layers in the SNN, as a heatmap.
    """
    if simulator is not None:
        assert simulator.record_weights, "Weight recording is not enabled."
        values = simulator.network.weights
        num_layers = len(simulator.network.synapse_layers)
    elif values is not None:
        values = values
        num_layers = len(values)
    else:
        raise ValueError("Either simulator with weight recording enabled or list of weight values for each synapse layer must be provided.")
    
    neurons_in = [w.shape[0] for w in values]
    neurons_out = [w.shape[1] for w in values]
    width = sum(neurons_out) * x_scale * 1.5
    height = max(neurons_in) * y_scale

    if figsize is None:
        figsize=(width, height)
    fig, axs = plt.subplots(1, num_layers, figsize=figsize, dpi=dpi,
                            squeeze=False, layout="constrained", gridspec_kw={"hspace": 0, "wspace": 0})
    wmin = min([np.min(w) for w in values])
    if bounded_weights:
        wmin = min(wmin, 0)
    wmax = max([np.max(w) for w in values])
    if bounded_weights:
        wmax = max(wmax, 1)
    
    for i in range(num_layers):
        ax = axs[0, i]
        img = ax.imshow(values[i], cmap=cmap, vmin=wmin, vmax=wmax, aspect=1.0, norm=color_scale)
        annotate_heatmap(img, valfmt="{x:.3f}", fontsize=10, textcolors=("white", "black"))
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_title(f"Synapse Layer {i}")
        ax.set_xlabel("Post-synaptic Neuron")
        ax.set_ylabel("Pre-synaptic Neuron")
        # Label action with names in last layer
        if env is not None and i == (num_layers - 1):
            num_actions = values[i].shape[1]
            axs.flat[-1].xaxis.set_ticks(ticks=range(num_actions), labels=[env.action_names[i] for i in range(num_actions)])
            ax.set_xlabel("Action")
        if env is not None and i == 0:
            ax.set_ylabel("State")

    fig.colorbar(img, ax=axs, orientation='vertical', label='Weight Value')
    fig.text(x=0.5, y=0.99, s="SNN Synaptic Weights", fontsize=20, ha='center')
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_weight_over_time(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, *, 
                          pre_id: int | List[int] = None, post_id: int | List[int] = None,
                          figsize=None, dpi: int = 100,
                          title: str = None, x_min=None, x_max=None,
                          synapse_layer: int = 0, 
                          savepath=None, show=True, line_kw={}):
    if simulator is not None:
        assert simulator.record_weights, "Weight recording is not enabled."
    values = simulator.weight_recorder.values if simulator is not None else values
    T = simulator.num_steps if simulator is not None else max([val.shape[2] for val in values])
    x_max = T if x_max is None else x_max
    x_min = 0 if x_min is None else x_min
    L = synapse_layer if synapse_layer < len(values) else 0
    dt = f"{simulator.dt} s" if simulator is not None else "1 unit"

    w_mat = values[L]
    if pre_id is not None:
        if not isinstance(pre_id, Sequence):
            pre_id = [pre_id]
        w_mat = np.take(w_mat, indices=pre_id, axis=0) # Slice along input neurons
    else:
        pre_id = [*range(w_mat.shape[0])]
    if post_id is not None:
        if not isinstance(post_id, Sequence):
            post_id = [post_id]
        w_mat = np.take(w_mat, indices=post_id, axis=1) # Slice along input neurons
    else:
        post_id = [*range(w_mat.shape[1])]

    nrow, ncol = w_mat.shape[:2]
    wmax = max(np.max(w_mat), 1)
    wmin = min(np.min(w_mat), 0)

    figsize=(5*ncol, 3*nrow) if figsize is None else figsize
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, dpi=dpi,
                            sharex=True, sharey=True, gridspec_kw={"hspace": 0, "wspace": 0},
                            squeeze=False)
    for i, pre in enumerate(pre_id):
        for j, post in enumerate(post_id):
            ax = axs[i, j]
            ax.plot(w_mat[i, j, :], **line_kw)
            ax.set_ylim(wmin, wmax)
            ax.set_xlim(x_min, x_max)
            if i == 0:
                ax.text(0.5, 1.05, f"Post Neuron {post}", transform=ax.transAxes, fontsize=16, ha="center")
            if j == ncol - 1:
                ax.text(1.05, 0.5, f"Pre Neuron {pre}", transform=ax.transAxes, fontsize=16, rotation=-90, va="center")
    plt.tight_layout()
    # plt.suptitle(title, y=0.9)
    # Labelling
    fig.text(s=f"Time ({dt})", fontsize=15, x=0.5, y=0.005, ha='center')
    fig.text(s="Weight", fontsize=15, ha='center', x=0.005, y=0.5, rotation=90, va='center')
    fig.text(s=title if title is not None else "SNN Weight over Time", fontsize=20, x=0.5, y=1.00, ha='center')

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_weight_heatmap(simulator: 'SNNSimulator', *, x_scale: float = 0.2, y_scale: float = 0.8,
                        synapse_layer: int = 0, t_min: int = None, t_max: int = None, t_range: int = 100,
                        log_scale: bool = False, cmap: str = "viridis",
                        figsize=None, dpi: int = 100,
                       savepath: str | Path = None, show: bool = True):
    if simulator.record_weights is False:
        raise ValueError("Weight recording is not enabled. Please enable it in the simulator configuration.")
    if t_max is None:
        t_max = simulator.num_steps
    if t_min is None:
        t_min = max(0, t_max - t_range)

    # Start plotting
    w_mat = simulator.weight_recorder.values[synapse_layer]
    num_outputs = simulator.network.output_size
    num_inputs = simulator.network.input_size
    if figsize is None:
        figsize = ((t_max - t_min) * x_scale, num_outputs * num_inputs * y_scale)
    fig, axs = plt.subplots(num_outputs, 1, figsize=figsize, dpi=dpi, sharex=True, gridspec_kw={"hspace": 0.0}, squeeze=False)
    axs: List[Axes]
    for i in range(num_outputs):
        ax = axs[i, 0]
        m = ax.imshow(w_mat[:, i, t_min:t_max], aspect='auto', cmap=cmap, norm=mpl.colors.LogNorm() if log_scale else None)
        ax.xaxis.set_ticks(np.arange(0, t_max - t_min, 10), labels= np.arange(t_min, t_max, 10))
        ax.set_ylabel(f"Neuron {i}", fontsize=12)
    axs[-1, 0].set_xlabel("Time Steps", fontsize=12)
    fig.colorbar(m, ax=axs, orientation='vertical', label='Weight Value')
    fig.text(0.5, 1.05, f"Weight Heatmap", fontsize=16)
    fig.text(0.5, 1.02, f"Synapse Layer {synapse_layer}", fontsize=12)
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_env_weight_actions(simulator: 'SNNSimulator', *, dpi: int = 100, figsize=None, comment: str = None,
                            savepath=None, show=True):
    # Extract objects
    network = simulator.network
    env = simulator.env

    if figsize is None:
        figsize=(env.width, env.height)
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("viridis").with_extremes(bad="white")

    w = network.weights[0]

    for i in range(4):
        ax = axs.flat[i]

        w_maze = env.maze.copy().astype(float)
        w_act = w[:, i].copy()
        emp_maze = np.zeros_like(w_maze)

        for state, pos in env._state_pos_dict.items():
            emp_maze[*pos] = w_act[state]

        ma_maze = np.ma.array(emp_maze, mask=(w_maze == 0))
        img = ax.imshow(ma_maze, extent=(0, env.width, 0, env.height), cmap=cmap, vmin=w.min(), vmax=w.max())

        ax.set_xticks(np.arange(0, env.width, 1), labels=[])
        ax.set_yticks(np.arange(0, env.height, 1), labels=[])
        ax.grid(visible=True, color='black', linewidth=0.7)
        ax.set_title(f"Action {env.action_names[i]}")
    plt.colorbar(img, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1, label="Weights")

    title = f"Overlaid SNN Weights by Action post-neuron"
    if comment is not None:
        title += "\n" + comment
    fig.text(0.5, 0.99, title, fontsize=12, ha='center')

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()


# def plot_env_weight_greedy(simulator: 'SNNSimulator', *, arrowcolors = ("white", "black"),
#                            threshold: float = 0.5,
#                            tolerance: float = 1e-10, dpi: int = 100, figsize=None, comment: str = None,
#                             savepath=None, show=True):
#     # Extract objects
#     network = simulator.network
#     env = simulator.env

    # if figsize is None:
    #     figsize=(env.width, env.height)
    # fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # cmap = plt.get_cmap("viridis").with_extremes(bad="white")

    # w = network.weights[0]

    # w_maze = env.maze.copy().astype(float)
    # w_act = w.max(axis=1)
    # emp_maze = np.zeros_like(w_maze)

    # for state, pos in env._state_pos_dict.items():
    #     emp_maze[*pos] = w_act[state]

    # ma_maze = np.ma.array(emp_maze, mask=(w_maze == 0))
    # img = ax.imshow(ma_maze, extent=(0, env.width, 0, env.height), cmap=cmap)

    # # Normalize the threshold to the images color range.
    # threshold = img.norm(ma_maze.max())*np.clip(threshold, 0, 1)

    # for state, pos in env._state_pos_dict.items():
    #     cnt = pos + 0.5
    #     act_vals = w[state, :]
    #     v_max = act_vals.max()
    #     for act, val in enumerate(act_vals):
    #         if np.abs(val - v_max) < tolerance:
    #             direction = env.action_map[act] * 0.5
    #             color = arrowcolors[int(img.norm(ma_maze[*pos]) > threshold)]
    #             ax.annotate('', xy=(cnt[1]+direction[1], env.height - (cnt[0]+direction[0])), xytext=(cnt[1], env.height - cnt[0]), 
    #                         arrowprops=dict(arrowstyle="->", edgecolor=color))

    # ax.set_xticks(np.arange(0, env.width, 1), labels=[])
    # ax.set_yticks(np.arange(0, env.height, 1), labels=[])
    # ax.grid(visible=True, color='black', linewidth=0.7)

    # title = f"Overlaid SNN Weights for each input state (maximum across Action neurons)"
    # if comment is not None:
    #     title += "\n" + comment
    # ax.set_title(title)
    # plt.colorbar(img, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1, label="Weights")

    # if savepath is not None:
    #     print(f"Saving plot to {savepath}")
    #     plt.savefig(savepath, dpi=dpi)
    # if show:
    #     plt.show()


def plot_env_weight_greedy(simulator: 'SNNSimulator', *, 
                            savepath=None, show=True, *kwargs):
    # Extract objects
    network = simulator.network
    env = simulator.env

    if env.obs_type == "state":
        # Single-synapselayer, state-based observation
        w = network.weights[0]
        state_act_vals = {state: w[state, :] for state in range(env._num_state)}
        _plot_env_action_values(env, state_act_vals, colour_code=True, savepath=savepath, show=show, 
                                legend_label="Weights", title="SNN SynapseLayer 0 weights overlaid on Environment",
                                **kwargs)
    elif env.obs_type == "position":
        encoder = simulator.spike_coder.encoder
        state_act_vals = _query_SNN_membrane_from_env_position(network, encoder, env)
        _plot_env_action_values(env, state_act_vals, savepath=savepath, show=show, 
                                legend_label="Membrane Potential", title="SNN Output Layer Membrane Potential for each Env state"
                                **kwargs)


def _query_SNN_membrane_from_env_position(network: 'SNN', encoder: SpikeCoder, env: 'BaseMaze') -> Dict[int, np.ndarray]:
    layer_mems = {}

    for state, pos in env._state_pos_dict.items():
        network.soft_reset()
        inp_buffer = encoder.generate_spikes(pos)
        spk_in = inp_buffer[:, 0]
        spk_out = network.forward(spk_in)

        layer_mems[state] = [n.membrane.copy() for n in network.neuron_layers[1:]]

    state_act_vals = {st: mem[-1] for st, mem in layer_mems.items()}
    return state_act_vals


def _plot_env_action_values(env: 'BaseMaze', state_action_values: Dict[int, np.ndarray], colour_code: bool = False, *, arrowcolors = ("white", "black"),
                           threshold: float = 0.5, legend_label: str = "Value", title: str = None,
                           tolerance: float = 1e-10, dpi: int = 100, figsize=None, comment: str = None,
                            savepath=None, show=True):

    if figsize is None:
        figsize=(env.width, env.height)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("viridis").with_extremes(bad="white")

    w_maze = env.maze.copy().astype(float)
    emp_maze = np.zeros_like(w_maze)

    if colour_code:
        for state, pos in env._state_pos_dict.items():
            emp_maze[*pos] = state_action_values[state].max()

    ma_maze = np.ma.array(emp_maze, mask=(w_maze == 0))
    img = ax.imshow(ma_maze, extent=(0, env.width, 0, env.height), cmap=cmap)

    # Normalize the threshold to the images color range.
    threshold = img.norm(ma_maze.max())*np.clip(threshold, 0, 1)

    for state, pos in env._state_pos_dict.items():
        cnt = pos + 0.5
        act_vals = state_action_values[state]
        v_max = act_vals.max()
        for act, val in enumerate(act_vals):
            if np.abs(val - v_max) < tolerance:
                direction = env.action_map[act] * 0.5
                color = arrowcolors[int(img.norm(ma_maze[*pos]) > threshold)]
                ax.annotate('', xy=(cnt[1]+direction[1], env.height - (cnt[0]+direction[0])), xytext=(cnt[1], env.height - cnt[0]), 
                            arrowprops=dict(arrowstyle="->", edgecolor=color))

    ax.set_xticks(np.arange(0, env.width, 1), labels=[])
    ax.set_yticks(np.arange(0, env.height, 1), labels=[])
    ax.grid(visible=True, color='black', linewidth=0.7)

    title = f"State-Action Potentials and Greedy Action" if title is None else title
    if comment is not None:
        title += "\n" + comment
    ax.set_title(title)
    plt.colorbar(img, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1, label=legend_label)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()


def plot_eligibility_traces(simulator: 'SNNSimulator' = None, values: np.ndarray = None, *, 
                            synapse_layer: int = 0, etype: Literal["pre", "post", "stdp", "custom"] = "pre",
                            x_scale: float = 0.2, y_scale: float = 0.8,
                            t_min: int = None, t_max: int = None, t_range: int = 100,
                            cmap: str = "viridis", figsize=None, dpi: int = 100,
                            savepath: str | Path = None, show: bool = True):
    if simulator is not None:
        if etype == "pre":
            assert simulator.record_eligibility_pre, "Pre-synaptic eligibility trace recording is not enabled."
            etrace = simulator.eligibility_pre_recorder.values[synapse_layer]
        elif etype == "pre":
            assert simulator.record_eligibility_post, "Post-synaptic eligibility trace recording is not enabled."
            etrace = simulator.eligibility_post_recorder.values[synapse_layer]
        elif etype == "stdp":
            assert simulator.record_eligibility_stdp, "STDP eligibility trace recording is not enabled."
            etrace = simulator.eligibility_stdp_recorder.values[synapse_layer]
        elif etype == "custom":
            assert simulator.record_eligibility_custom, "Custom eligibility trace recording is not enabled."
            etrace = simulator.eligibility_custom_recorder.values[synapse_layer]
        else:
            raise ValueError(f"Eligiblity trace type: {etype} not supported.")
        
        num_inputs = simulator.network.input_size
        num_outputs = simulator.network.output_size
        num_steps = simulator.num_steps
    elif values is not None:
        etrace = values
        etype = None
        num_inputs = etrace.shape[0]
        num_outputs = etrace.shape[1]
        num_steps = etrace.shape[2]
    else:
        raise ValueError("Either a simulator (with eligibility trace recording enabled) or a numpy array of shape [num_inputs, num_outputs, num_steps], must be provided.")
    # if simulator.record_eligibility is False:
    #     raise ValueError("Eligibility trace recording is not enabled. Please enable it in the simulator configuration.")
    if t_max is None:
        t_max = num_steps
    if t_min is None:
        t_min = max(0, t_max - t_range)

    emin = np.min(etrace)
    emax = np.max(etrace)

    if figsize is None:
        figsize = ((t_max - t_min) * x_scale, num_outputs * num_inputs * y_scale)
    fig, axs = plt.subplots(num_outputs, 1, figsize=figsize, dpi=dpi, sharex=True, layout="constrained", gridspec_kw={"hspace": 0.0},
                            squeeze=False)

    for i, j in enumerate(reversed(range(num_outputs))):
        ax = axs[i, 0]
        m = ax.imshow(etrace[:, j, t_min:t_max], cmap=cmap, aspect='auto', origin="lower", vmin=emin, vmax=emax)
        ax.xaxis.set_ticks(np.arange(0, t_max - t_min, 10), labels= np.arange(t_min, t_max, 10))
        ax.set_ylabel(f'Neuron {j}')

    fig.colorbar(m, label='Eligibility Traces', ax=axs)
    axs[-1, 0].set_xlabel("Time steps")
    fig.suptitle(f"Eligibility Traces: {etype}\nSynapse Layer {synapse_layer}", fontsize=16)
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


def plot_intermediate_fitness(simulator: 'SNN_Simulator' = None, values: np.ndarray = None, *, plot_exploration: bool = False,
                              use_cutoff: bool = False,
                              num_steps: int = None, timestamps: np.ndarray = None,
                              x_scale: float = 0.01, y_scale: float = 1.0, x_eps: int = 1,
                              t_min: int = None, t_max: int = None, t_range: int = None, window_size: int = 10, 
                              figsize: tuple = None, dpi: int = 100,
                              savepath: str | Path = None, show: bool = True):
    if simulator is not None:
        fts = simulator.get_intermediate_fitness(use_cutoff=use_cutoff)
        ft = simulator.get_fitness()
        T = simulator.num_steps
        # eps_len = simulator.reward_collector.get_episode_lengths()
        # eps_timestamp = np.cumsum(eps_len) * simulator.spike_coder.input_delay
        eps_timestamp = simulator.get_episode_timestamps(use_cutoff=use_cutoff)
    elif values is not None:
        fts = values
        ft = np.mean(fts)
        T = num_steps if num_steps is not None else len(fts)
        eps_timestamp = None
        if num_steps is None:
            x_scale *= 100    

    if timestamps is None and eps_timestamp is None:
        ts = np.linspace(0, T, len(fts)) if num_steps is not None else np.arange(0, len(fts))
    elif eps_timestamp is not None:
        ts = eps_timestamp
    else:
        assert len(timestamps) == len(fts), "Timestamps must match the length of fitness values."
        ts = timestamps
    # ts = np.arange(simulator.spike_generator.pattern_length - 1, T, simulator.spike_generator.length)
    window_size = min(window_size, len(fts))
    runavg = np.convolve(fts, np.ones(window_size) / window_size, mode='same')

    if t_range is None:
        t_range = T
    if t_max is None:
        t_max = T
    if t_min is None:
        t_min = max(0, t_max - t_range)

    if figsize is None:
        figsize = (((t_max - t_min) * x_scale, 10 * y_scale * (int(plot_exploration) + 1)))
    fig, axs = plt.subplots(1 + int(plot_exploration), 1, figsize=figsize, dpi=dpi,
                            layout="constrained", squeeze=False)
    ax = axs[0, 0]
    ax.plot(
        ts, fts,
        color="gray", alpha=0.8,
        drawstyle="steps-pre",
        linewidth=1, label="Intermediate Fitness",
        marker="o", markersize=5
    )
    ax.plot(ts, runavg, color="blue", linewidth=2, label=f"Running Average ({window_size})", markersize=10, marker="o")
    ax.legend(loc="lower right", fontsize=12)
    if num_steps is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.set_xlim(t_min - x_eps, t_max + x_eps)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Episode " + simulator.reward_collector.fitness_type.title())
    if plot_exploration:
        ax = axs[1, 0]
        expl = simulator.reward_collector.get_explorations()
        ts = [r.t for r in simulator.reward_collector.records]
        ax.plot(
            ts, expl,
            color="red", alpha=0.8,
            drawstyle="steps-pre",
            linewidth=2.5, 
        )
        ax.set_xlim(t_min - x_eps, t_max + x_eps)
        if num_steps is not None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(100))
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Exploration Rate")
    fig.text(0.5, 1.07, "Intermediate Fitness Over Time", ha='center', fontsize=20)
    agg_func = simulator.reward_collector.fitness_agg_func
    fig.text(0.5, 1.02, f"{agg_func.title()} Fitness: {ft:.2f}", ha='center', fontsize=14)
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    # plt.close(fig)


### Plotting functions for Learning Rule ###

def plot_learning_rule(rule: 'base.LearningRule', simulator: 'SNNSimulator' = None, **kwargs):
    input_size = getattr(rule, "input_size")
    if input_size == 2:
        plot_learning_rule_2D(rule, simulator, **kwargs)
    elif input_size == 3:
        plot_learning_rule_3D(rule, simulator, **kwargs)
    elif input_size == 4:
        plot_learning_rule_4D(rule, simulator, **kwargs)
    else:
        raise ValueError(f"Plotting learning rule is not yet supported for input_size={input_size}")


def plot_learning_rule_4D(rule: 'base.LearningRule', simulator: 'SNNSimulator' = None, *, 
                          custom_bounds: dict[str, tuple] = None,
                          n_bins: int = 100, n_cols: int = 5, n_rows: int = 5,
                          var_col: str = None, var_row: str = None,
                          var_x: str = None, var_y: str = None,
                          cmap: str = "RdBu", aspect="auto",
                          rule_name: str = None, title: str = None,
                          figsize: tuple = (20, 20), dpi: int = 100,
                          savepath: str | Path = None, show: bool = True,
                          **kwargs) -> None:
    # DONE: Fix format to be generic like plot_learning_rule_1D
    # Use default values for each inputs
    if custom_bounds is not None:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) if inp not in custom_bounds else custom_bounds.get(inp) for inp in rule.input_order]
    else:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) for inp in rule.input_order]
    # TODO: Update bounds with recorded values if a Simulator is passed in

    # Input Validation
    var_names: List = rule.input_order.copy()
    # First, if any variable is specified, remove them from variable list
    if var_x is not None:
        assert var_x in rule.input_order, f"X-Axis Variable {var_x} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_x = var_names.pop(var_names.index(var_x))
    if var_y is not None:
        assert var_y in rule.input_order, f"Y-Axis Variable {var_y} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_y = var_names.pop(var_names.index(var_y))
    if var_col is not None:
        assert var_col in rule.input_order, f"Column Variable {var_col} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_col = var_names.pop(var_names.index(var_col))
    if var_row is not None:
        assert var_row in rule.input_order, f"Row Variable {var_row} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_row = var_names.pop(var_names.index(var_row))
    # Then for any variable not specified, use them by order of (x, y, col, row)
    if var_x is None:
        var_x = var_names.pop(0)
    if var_y is None:
        var_y = var_names.pop(0)
    if var_col is None:
        var_col = var_names.pop(0)
    if var_row is None:
        var_row = var_names.pop(0)

    # Get index position for each variable name
    i_x = rule.input_order.index(var_x)
    i_y = rule.input_order.index(var_y)
    i_col = rule.input_order.index(var_col)
    i_row = rule.input_order.index(var_row)

    # Set axis names for column and remaining x-y variables
    # var_col = var_names.get(i_col)
    # var_row = var_names.get(i_row)


    # Index for x-y variables
    # xy_index = [i for i in range(rule.input_size) if i not in (i_col, i_row)]
    # i_x, i_y = xy_index

    # Set extent to be used with imshow for remaining x-y variables
    xy_bounds = [bounds[i_x], bounds[i_y]]
    xy_bounds = [x for xy in xy_bounds for x in xy]

    # Preparing spaces for each variables
    num_meshes = []
    for i in range(rule.input_size):
        if i == i_col:
            num_meshes.append(n_cols)
        elif i == i_row:
            num_meshes.append(n_rows)
        else:
            num_meshes.append(n_bins)

    bins = [np.linspace(low, upp, n) for (low, upp), n in zip(bounds, num_meshes)]

    # Query the rule for each value of column variable
    # The reason this has to be done in a for loop is because array.reshape
    #   does not guarantee accurate values when i_col is not the last var
    vv = np.empty(shape=(n_bins, n_bins, n_cols, n_rows)) # The reason this order is fixed is for easier formatting and since
    #                                                        it doesn't depend on exact rule ordering anymore

    for xj in range(n_rows):
        for xi in range(n_cols): 
            # Make meshgrid based on remaining x-y variables
            xx, yy = np.meshgrid(bins[i_x], bins[i_y], indexing='xy')
            # Make a constant array broadcasted to 2D grid
            cc = np.full([n_bins, n_bins], bins[i_col][xi])
            rr = np.full((n_bins, n_bins), bins[i_row][xj])
            # Insert this z-array into the position governed by i_col
            inp_list = [None, None, None, None]
            inp_list[i_row] = rr
            inp_list[i_col] = cc
            inp_list[i_x] = xx
            inp_list[i_y] = yy
            # Flatten inputs and query learning rule
            inp = np.concatenate([ip.reshape(-1, 1) for ip in inp_list], axis=-1)
            outp = rule.forward(inp)
            vv[:, :, xi, xj] = outp.reshape((n_bins, n_bins))

    # After all values are queried, build a divergent colormap based centered at 0 
    #   and global min-max of outputs
    vmin = vv.min()
    vmax = vv.max()
    colorizer = mpl.colorizer.Colorizer(cmap=cmap, norm=mpl.colors.CenteredNorm(vcenter=0))
    colorizer.set_clim(vmin, vmax)

    # Values are ready, plots can be made
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    for xj in range(n_rows):
        for xi in range(n_cols):
            ax: Axes = axs[xj, xi]
            val_xi = bins[i_col][xi]
            val_xj = bins[i_row][xj]
            img = ax.imshow(vv[:, :, xi, xj], extent=xy_bounds, aspect=aspect, origin="lower", colorizer=colorizer)
            ax.set_box_aspect(1)
            if xj == 0:
                ax.text(0.5, 1.05, f"{var_col} = {val_xi:.2f}", fontsize=15, ha="center") # column label
            if xi == n_cols - 1:
                ax.text(1.05, 0.5, f"{var_row} = {val_xj:.2f}", rotation=90, fontsize=15, va="center") # row label

    fig.text(0.51, 0.22, var_x, fontsize=20, ha="center", transform=fig.transFigure) # xlabel
    fig.text(0.1, 0.55, var_y, fontsize=20, rotation=90, va="center", transform=fig.transFigure) # ylabel

    # Optionally display rule_path or name
    if rule_name is not None:
        comment = "Rule: " + str(rule_name)
        fig.text(0.99, 0.1, comment, ha="right", transform=fig.transFigure, fontsize=15)
    # Title
    title = "Learning Rule Response to Inputs:\n" + f"f({rule.input_order}) = ΔWeight" if title is None else title
    fig.text(0.5, 0.92, title, ha="center", transform=fig.transFigure, fontsize=20)
    # Colorbar for delta weight
    cbar = fig.colorbar(img, ax=axs, fraction=0.05, orientation="horizontal", aspect=100)
    cbar.set_label(label='ΔWeight', size=20)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_learning_rule_3D(rule: 'base.LearningRule', simulator: 'SNNSimulator' = None, *, 
                          custom_bounds: dict[str, tuple] = None,
                          n_bins: int = 100, n_cols: int = 5, 
                          var_x: str = None, var_y: str = None, var_col: str = None,
                          cmap: str = "RdBu", figsize: tuple = (20, 5), aspect="auto", dpi: int = 100,
                          rule_name: str = None, title: str = None,
                          savepath: str | Path = None, show: bool = True,
                          **kwargs) -> None:
    # DONE: Fix format to be generic like plot_learning_rule_1D
    # Use default values for each inputs
    if custom_bounds is not None:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) if inp not in custom_bounds else custom_bounds.get(inp) for inp in rule.input_order]
    else:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) for inp in rule.input_order]
    # TODO: Update bounds with recorded values if a Simulator is passed in

    # # Set axis names for column and remaining x-y variables
    # assert var_col in rule.input_order, f"Variable {var_col} to set by column must exists within rule." + \
    #     f" Rule only uses following inputs: {rule.input_order}"
    # var_names = rule.input_order.copy()
    # i_col = var_names.index(var_col)
    # var_col = var_names.pop(i_col)

    # Input Validation
    var_names: List = rule.input_order.copy()
    # First, if any variable is specified, remove them from variable list
    if var_x is not None:
        assert var_x in rule.input_order, f"X-Axis Variable {var_x} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_x = var_names.pop(var_names.index(var_x))
    if var_y is not None:
        assert var_y in rule.input_order, f"Y-Axis Variable {var_y} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_y = var_names.pop(var_names.index(var_y))
    if var_col is not None:
        assert var_col in rule.input_order, f"Column Variable {var_col} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_col = var_names.pop(var_names.index(var_col))

    # Then for any variable not specified, use them by order of (x, y, col)
    if var_x is None:
        var_x = var_names.pop(0)
    if var_y is None:
        var_y = var_names.pop(0)
    if var_col is None:
        var_col = var_names.pop(0)

    # Get index position for each variable name
    i_x = rule.input_order.index(var_x)
    i_y = rule.input_order.index(var_y)
    i_col = rule.input_order.index(var_col)

    # Set extent to be used with imshow for remaining x-y variables
    # xy_bounds = bounds.copy()
    # xy_bounds.pop(i_col)
    # xy_bounds = [x for xy in xy_bounds for x in xy]
    xy_bounds = [bounds[i_x], bounds[i_y]]
    xy_bounds = [x for xy in xy_bounds for x in xy]

    # Index for x-y variables
    # xy_index = [*range(rule.input_size)]
    # xy_index.pop(i_col)

    # Preparing spaces for each variables
    # num_meshes = [n_bins, n_bins]
    # num_meshes.insert(i_col, n_cols)
    num_meshes = []
    for i in range(rule.input_size):
        if i == i_col:
            num_meshes.append(n_cols)
        else:
            num_meshes.append(n_bins)

    bins = [np.linspace(low, upp, n) for (low, upp), n in zip(bounds, num_meshes)]

    # Query the rule for each value of column variable
    # The reason this has to be done in a for loop is because array.reshape
    #   does not guarantee accurate values when i_col is not the last var
    vv = np.empty((n_bins, n_bins, n_cols))

    for xi in range(n_cols):
        # Make meshgrid based on remaining x-y variables
        xx, yy = np.meshgrid(bins[i_x], bins[i_y], indexing='xy')
        # Make a constant array broadcasted to 2D grid
        z = bins[i_col][xi]
        zz = np.full([n_bins, n_bins], z)
        # Insert this z-array into the position governed by i_col
        inp_list = [xx, yy]
        inp_list.insert(i_col, zz)
        # Flatten inputs and query learning rule
        inp = np.concatenate([ip.reshape(-1, 1) for ip in inp_list], axis=-1)
        outp = rule.forward(inp)
        vv[:, :, xi] = outp.reshape((n_bins, n_bins))

    # After all values are queried, build a divergent colormap based centered at 0 
    #   and global min-max of outputs
    vmin = vv.min()
    vmax = vv.max()
    colorizer = mpl.colorizer.Colorizer(cmap=cmap, norm=mpl.colors.CenteredNorm(vcenter=0))
    colorizer.set_clim(vmin, vmax)

    # Values are ready, plots can be made
    fig, axs = plt.subplots(1, n_cols, figsize=figsize, dpi=dpi)
    for xi in range(n_cols):
        ax: Axes = axs[xi]
        val_xi = bins[i_col][xi]
        img = ax.imshow(vv[:, :, xi], extent=xy_bounds, aspect=aspect, origin="lower", colorizer=colorizer, **kwargs)
        ax.set_box_aspect(1)
        ax.set_title(f"{var_col} = {val_xi}")
        if xi == 0:
            ax.set_xlabel(var_x)
            ax.set_ylabel(var_y)

    # Optionally display rule_path or name
    if rule_name is not None:
        comment = "Rule: " + str(rule_name)
        fig.text(0.99, 0.1, comment, ha="right", transform=fig.transFigure, fontsize=15)
    # Title
    title = "Learning Rule Response to Inputs:\n" + f"f({rule.input_order}) = ΔWeight" if title is None else title
    fig.text(0.5, 0.92, title, ha="center", transform=fig.transFigure, fontsize=20)
    # Colorbar for delta weight
    cbar = fig.colorbar(img, ax=axs, fraction=0.05, orientation="horizontal", aspect=100)
    cbar.set_label(label='ΔWeight', size=20)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_learning_rule_2D(rule: base.LearningRule, simulator: 'SNNSimulator' = None, *, 
                          custom_bounds: dict[str, tuple] = None,
                          n_bins: int = 100, var_x: str = None, var_y: str = None, 
                          cmap: str = "RdBu", figsize: tuple = (10, 10), dpi: int = 100,
                          rule_name: str = None, title: str = None,
                          savepath: str | Path = None, show: bool = True,
                          **kwargs):
    """
    Plot Learning Rule in a single heatmap, with outputs as color values.
    First variable in `input_order` will be on x-axis, second variable on y-axis (unless `transpose=True`)

    Args:
        rule (base.LearningRule): Learning Rule to be plotted
        simulator (SNNSimulator, optional): If a Simulator is passed in, boundaries for each input variable will be extracted. Defaults to None.
        n_bins (int, optional): Level of granularity in plot. Defaults to 100.
        cmap (str, optional): Name of matplotlib color map. Defaults to "RdBu".
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        savepath (str | Path, optional): Path to save the figure. If None, will not save the plot. Defaults to None.
        show (bool, optional): Whether to call `plt.show()` at the end. Defaults to True.
    """
    if custom_bounds is not None:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) if inp not in custom_bounds else custom_bounds.get(inp) for inp in rule.input_order]
    else:
        bounds = [LRULE_INPUT_BOUNDS.get(inp) for inp in rule.input_order]
    # TODO: Update boundaries if simulator is passed in
        
    # Input Validation
    var_names: List = rule.input_order.copy()
    # First, if any variable is specified, remove them from variable list
    if var_x is not None:
        assert var_x in rule.input_order, f"X-Axis Variable {var_x} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_x = var_names.pop(var_names.index(var_x))
    if var_y is not None:
        assert var_y in rule.input_order, f"Y-Axis Variable {var_y} must exists within rule." + \
            f" Rule only uses following inputs: {rule.input_order}"
        var_y = var_names.pop(var_names.index(var_y))
    # Then for any variable not specified, use them by order of (x, y)
    if var_x is None:
        var_x = var_names.pop(0)
    if var_y is None:
        var_y = var_names.pop(0)

    # Get index position for each variable name
    i_x = rule.input_order.index(var_x)
    i_y = rule.input_order.index(var_y)

    # Set xy extents
    xy_bounds = [bounds[i_x], bounds[i_y]]
    xy_bounds = [x for xy in xy_bounds for x in xy]

    # Create bins
    num_meshes = [n_bins, n_bins]
    bins = [np.linspace(low, upp, n) for (low, upp), n in zip(bounds, num_meshes)]

    # Query the rule for each value of column variable
    vv = np.empty(shape=(n_bins, n_bins)) 

    # Make meshgrid based on remaining x-y variables
    xx, yy = np.meshgrid(bins[i_x], bins[i_y], indexing='xy')
    # Insert this z-array into the position governed by i_col
    inp_list = [None, None]
    inp_list[i_x] = xx
    inp_list[i_y] = yy
    # Flatten inputs and query learning rule
    inp = np.concatenate([ip.reshape(-1, 1) for ip in inp_list], axis=-1)
    outp = rule.forward(inp)
    vv[:, :] = outp.reshape((n_bins, n_bins))

    # After all values are queried, build a divergent colormap based centered at 0 
    #   and global min-max of outputs
    vmin = vv.min()
    vmax = vv.max()
    colorizer = mpl.colorizer.Colorizer(cmap=cmap, norm=mpl.colors.CenteredNorm(vcenter=0))
    colorizer.set_clim(vmin, vmax)

    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False, dpi=dpi)

    ax: Axes = axs[0, 0]
    # extents = [xmin, xmax, ymin, ymax] if not transpose else [ymin, ymax, xmin, xmax]
    img = ax.imshow(vv, extent=xy_bounds, origin='lower', aspect='auto', cmap=cmap, norm=mpl.colors.CenteredNorm(), **kwargs)
    ax.set_box_aspect(1)

    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)

    # Optionally display rule_path or name
    if rule_name is not None:
        comment = "Rule: " + str(rule_name)
        fig.text(0.99, 0.1, comment, ha="right", transform=fig.transFigure, fontsize=15)
    # Title
    title = "Learning Rule Response to Inputs:\n" + f"f({rule.input_order}) = ΔWeight" if title is None else title
    fig.text(0.5, 0.92, title, ha="center", transform=fig.transFigure, fontsize=20)
    # Colorbar for delta weight
    cbar = fig.colorbar(img, ax=axs, fraction=0.05, orientation="horizontal", aspect=100)
    cbar.set_label(label='ΔWeight', size=20)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

### Plotting functions for Evolutionary Results ###


def plot_fitness_generation(file_path: str | Path = None, res: pd.DataFrame = None, *, 
                            x_var: str = "gen", y_var: str = "avg_fitness",
                            estimator: str = "mean", errorband: str | tuple = ("pi", 100),
                            hue_var: str = None, run_name: str = None, merge_avg: bool = False,
                            linecolor_best: str = "black", linecolor_est: str = "blue", pointcolor: str = "gray",
                            sns_style: str = "whitegrid", sns_palette: str = "muted", figsize: tuple = None, dpi: int = 100,
                            legend: bool = True, fontscale: float = 1.0,
                            title: str = None, subtitle: str = None, comment: str = None,
                            x_eps: int = 2, x_scale: float = 0.3, y_scale: float = 1.3, y_size: float = 10,
                            ymin: float = None, ymax: float = None,
                            savepath: str | Path = None, show: bool = True):
    if file_path is not None:
        assert os.path.exists(file_path), f"File {file_path} does not exist."
        res = pd.read_csv(f"{file_path}")
        run_name = run_name if run_name is not None else Path(file_path).parent.stem
    if res is not None:
        res = res
        run_name = None if run_name is None else run_name

    assert x_var in res.columns, f"x-axis variable: {x_var} not present in DataFrame columns"
    assert y_var in res.columns, f"y-axis variable: {y_var} not present in DataFrame columns"
    if hue_var is not None:
        assert hue_var in res.columns, f"Hue variable to plot ({hue_var}) must be in DataFrame."

    # Calulate best all-time fitness
    best_fts = res.groupby(x_var)[y_var].max().cummax()
    best_fts.rename("best_fitness", inplace=True)
    res = res.join(best_fts, on=x_var, how="left")

    num_gens = res[x_var].max() + 1
    best_fts = res["best_fitness"].max()
    ymax = res[y_var].max() if ymax is None else float(ymax)
    ymin = res[y_var].min() if ymin is None else float(ymin)

    if figsize is None:
        figsize=(num_gens * x_scale, y_size) 
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # sns.set_theme(palette=sns_palette, style=sns_style)
    # Fitness per individual
    if hue_var is None:
        sns.stripplot(data=res, x=x_var, y=y_var, size=5, ax=ax, alpha=0.5, color=pointcolor)
    else:
        sns.stripplot(data=res, x=x_var, y=y_var, hue=hue_var, size=5, ax=ax, alpha=0.5, palette=sns_palette)
    # Best cumulative fitness
    sns.lineplot(data=res, x=x_var, y="best_fitness", color=linecolor_best, linewidth=2, ax=ax, label="Best Fitness")
    # Average fitness for each generation
    if hue_var is None or merge_avg:
        sns.lineplot(data=res, x=x_var, y=y_var, estimator=estimator, errorbar=errorband, ax=ax,
                    color=linecolor_est, linewidth=2, label=f"{estimator.title()} Fitness")
    else:
        sns.lineplot(data=res, x=x_var, y=y_var, estimator=estimator, errorbar=errorband, ax=ax, hue=hue_var,
                    palette=sns_palette, linewidth=2)
    
    ax.set_xlim(0-x_eps, num_gens+x_eps)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel("Generation", fontsize=12*fontscale)
    ax.set_ylabel("Fitness", fontsize=12*fontscale)
    if legend:
        ax.legend(loc="upper left", fontsize=12*fontscale)
    else:
        ax.get_legend().remove()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.text(num_gens - 1, best_fts, f"{best_fts:.2f}", ha='right', va="bottom", fontsize=16*fontscale, transform=ax.transData, color=linecolor_best)
    title_main = f"Fitness Over Generations" if title is None else title
    subtitle = f"({estimator.title()} Fitness +/- {errorband[1]} {errorband[0].upper()})" if subtitle is None else subtitle
    comment = f"Run: {run_name}" if comment is None else comment
    fig.text(0.5, 0.95, title_main, ha='center', fontsize=24*fontscale)
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=16*fontscale)
    ax.text(1.00, 1.05, comment, ha='right', fontsize=16*fontscale, transform=ax.transAxes)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_solution_generation(solution_file: str | Path = None, var: Literal["global_fitness", "local_fitness", "novelty_dist", "rank"] = "global_fitness", df: pd.DataFrame = None, *, 
                             estimator: str = "mean", errorband: str | tuple = ("pi", 100),
                            linecolor_best: str = "black", linecolor_est: str = "blue", point_cmap: str = 'dark:gray',
                            sns_style: str = "whitegrid", sns_palette: str = "muted", figsize: tuple = None, dpi: int = 100,
                            title: str = None, subtitle: str = None, comment: str = None,
                            x_eps: int = 2, x_scale: float = 0.3, y_scale: float = 1.3, y_size: float = 10,
                            savepath: str | Path = None, show: bool = True):
    # assert var in ["global_fitness", "local_fitness", "novelty_dist", "rank"]

    if solution_file is not None:
        run_name = Path(solution_file).parent.stem

        # Read data
        solutions = []
        with open(solution_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                solutions.append(row)

        sols_df = pd.DataFrame.from_records(solutions, exclude=["genome", "behaviour"], coerce_float=True)
        sols_df = sols_df.astype({'gen': 'int', 'rank': 'int', 'global_fitness': 'float', 'local_fitness': 'float', 'novelty_dist': 'float'})
        sols_df["indiv_type"] = sols_df["indiv"].str.slice(0, 1).map({"p":"Parent", "o": "Offspring"})
        assert var in sols_df.columns, f"var={var} not found in solution file columns: {sols_df.columns}"
        df = sols_df
    elif df is not None:
        _cols_to_check = ["gen", "indiv", var]
        _check_cols = np.isin(_cols_to_check, df.columns)
        assert np.all(_check_cols), f"DataFrame is missing columns: {np.asarray(_cols_to_check)[~_check_cols]}"
        if "indiv_type" not in df.columns:
            df["indiv_type"] = df["indiv"].str.slice(0, 1).map({"p":"Parent", "o": "Offspring"})
        df = df
        run_name = "NA"
    else:
        raise AssertionError(f"Either 'solution_file' or 'df' must be specified.")

    # Calulate best all-time fitness
    best_fts = df.groupby("gen")[var].max().cummax()
    best_fts.rename("best_fitness", inplace=True)
    res = df.join(best_fts, on="gen", how="left")

    num_gens = res["gen"].max() + 1
    best_fts = res["best_fitness"].max()
    # fts_range = res[var].max() - res[var].min()
    var_pretty = var.replace('_', ' ').title()

    fig, ax = plt.subplots(1, 1, figsize=(num_gens * x_scale, y_size) if figsize is None else figsize, dpi=dpi)
    # sns.set_theme(palette=sns_palette, style=sns_style)
    # Fitness per individual
    sns.stripplot(data=res, x="gen", y=var, hue="indiv_type", size=5, ax=ax, alpha=0.5, palette=point_cmap)
    # Best cumulative fitness
    sns.lineplot(data=res, x="gen", y="best_fitness", color=linecolor_best, linewidth=2, ax=ax, label=f"Best {var_pretty}")
    # Average fitness for each generation
    sns.lineplot(data=res, x="gen", y=var, estimator=estimator, errorbar=errorband, ax=ax, color=linecolor_est, linewidth=2, label=f"{estimator.title()} {var_pretty}")
    ax.set_xlim(0-x_eps, num_gens+x_eps)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(var_pretty, fontsize=12)
    # ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.text(num_gens - 1, best_fts, f"{best_fts:.2f}", ha='right', va="bottom", fontsize=16, transform=ax.transData, color=linecolor_best)
    title_main = f"{var_pretty} Over Generations" if title is None else title
    subtitle = f"({estimator.title()}  +/- {errorband[1]} {errorband[0].upper()})" if subtitle is None else subtitle
    comment = f"Run: {run_name}" if comment is None else comment
    fig.text(0.5, 0.95, title_main, ha='center', fontsize=24)
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=16)
    ax.text(1.00, 1.05, comment, ha='right', fontsize=16, transform=ax.transAxes)

    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return df

## Plotting functions for entire experiment (containing multiple runs) ##
def plot_compare_run(exp_dir: str | Path, x_var: str, hue_var: str = None, col_var: str = None, *,
                     savepath: str | Path = None, show: bool = True):
    exp_path = Path(exp_dir)
    eval_file = exp_path / "eval_result.csv"
    assert eval_file.exists(), f"Evaluation results file {eval_file} does not exist."

    # Load previously evaluated results
    cfgs = pd.read_csv(eval_file)
    num_evals = cfgs["num_evals"].unique()[0]
    num_sim_steps = cfgs["num_sim_steps"].unique()[0]
    exp_name = exp_path.name

    # Create the bar plot
    g = sns.catplot(
        data=cfgs,
        x=x_var,
        y="mean_fts",
        hue=hue_var,
        col=col_var,
        kind="bar",
        errorbar=None, 
        height=8,
        aspect=1.2,
        sharey=False,
    )
    fig = g.figure

    # Add error bars manually using std_fts
    if col_var is not None:
        for ax, col_val in zip(g.axes.flat, cfgs[col_var].unique()):
            subset = cfgs[cfgs[col_var] == col_val].sort_values(by=[hue_var, x_var])
            # ax = g.axes[0, 0]  # Get the first axis
            # subset = cfgs.sort_values(by=[hue_var, x_var])
            for i, bar in enumerate(ax.patches):
                if i >= len(subset):
                    continue
                # Get the corresponding std_fts value
                std_err = subset.iloc[i]["std_fts"]
                # Add error bar
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    yerr=std_err,
                    fmt="none",
                    c="gray",
                    capsize=3,
                    elinewidth=1,
                )
                ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    else:
        ax = g.axes[0, 0]  # Get the first axis
        subset = cfgs.groupby([hue_var, x_var])["std_fts"].mean().reset_index().sort_values(by=[hue_var, x_var])
        for i, bar in enumerate(ax.patches):
            if i >= len(subset):
                continue
            # Get the corresponding std_fts value
            std_err = subset.iloc[i]["std_fts"]
            # Add error bar
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=std_err,
                fmt="none",
                c="gray",
                capsize=3,
                elinewidth=1,
            )
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    # Adjust the plot

    if col_var is not None:
        g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels(x_var.replace("_", " ").title(), "Fitness")
    g.legend.set_title(hue_var.replace("_", " ").title())
    plt.text(0.5, 1.1, " -- ".join([var.replace("_", " ").title() for var in [x_var, hue_var, col_var] if var is not None]),
            transform=fig.transFigure, ha='center', fontsize=20)
    plt.text(0.5, 1.05, f"{exp_name} | {num_sim_steps} Simulation timesteps", transform=fig.transFigure, ha='center', fontsize=16)
    plt.text(0.5, 1.02, f"Mean Fitness +/- Std ({num_evals} Evaluations)", transform=fig.transFigure, ha='center', fontsize=12)
    if savepath is not None:
        print(f"Saving plot to {savepath}")
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    # plt.close(fig)


### Helper functions ###

def _plot_membrane_old(membrane_array, ax: Axes, threshold=None, title=None):
    ax.plot(membrane_array)
    if threshold is not None:
        ax.axhline(threshold, color='gray', linestyle='--')
    if title is not None:
        ax.set_title(title)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Taken from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_runs(simulators: List['SNNSimulator'], plot_spikes=True, plot_traces=True, plot_membranes=True, plot_weights=True, plot_weights_time=True, 
              x_min=0, x_max=None, y_scale=0.5, x_scale=0.2, save_path: str | Path = None, variable: str = "tau_trace"):
        
    for i, simulator in enumerate(simulators):
        # org_tau_trace = tau_traces[i]
        org_tau_trace = int(simulator.network.neuron_params[-1].get(variable) / simulator.dt)
        # Plotting
        if plot_spikes:
            plot_spikes(simulator, title=f"Spike Train\n{variable}={org_tau_trace} dt", cmap="viridis", cmap_range=(0, 0.8), x_eps=0.0, x_min=x_min, x_max=x_max, y_scale=y_scale, x_scale=x_scale,
                                savepath=Path(save_path) / f"{variable}_{org_tau_trace}_spikes.png" if save_path else None)
        if plot_traces:
            plot_traces(simulator, title=f"Pre- and Post-Synaptic Neuron Trace\n{variable}={org_tau_trace} dt", cmap="viridis", cmap_range=(0, 0.8), x_min=x_min, x_max=x_max, y_scale=y_scale, x_scale=x_scale,
                                savepath=Path(save_path) / f"{variable}_{org_tau_trace}_traces.png" if save_path else None)
        if plot_membranes:
            plot_membranes(simulator, title=f"Membrane Potentials\n{variable}={org_tau_trace} dt", plot_inputs=False, cmap="viridis", cmap_range=(0, 0.8), col_width=20, row_height=3, x_min=x_min, x_max=x_max,
                                   savepath=Path(save_path) / f"{variable}_{org_tau_trace}_membranes.png" if save_path else None)
        if plot_weights:
            plot_weights(simulator, title=f"Weight Matrices\n{variable}={org_tau_trace} dt", div=8,
                                 savepath=Path(save_path) / f"{variable}_{org_tau_trace}_weights.png" if save_path else None)
        if plot_weights_time:
            plot_weight_over_time(simulator, title=f"Weight change\n{variable}={org_tau_trace} dt",
                                          savepath=Path(save_path) / f"{variable}_{org_tau_trace}_weights_time.png" if save_path else None)
            

if __name__ == "__main__":
    fname = "simulators_05-16-18-56_on-spike_BinaryClass.pkl"
    simulators = pickle.load(open(Path("data", fname), "rb"))
    T = simulators[0].num_steps
    parent = "plots" / Path(fname).with_suffix("")
    parent.mkdir(exist_ok=True, parents=True)
    # Plotting parameters
    x_min = 0
    x_max = T
    plot_runs(simulators, plot_spikes=True, plot_traces=True, plot_membranes=True, plot_weights=True, plot_weights_time=True, x_min=x_min, x_max=x_max, x_scale=(x_max-x_min)/1500, y_scale=0.5, save_path=parent)