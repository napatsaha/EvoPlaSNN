from pathlib import Path
import pickle
from typing import List, Literal
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

### Plotting functions that require Simulator ###


def plot_spikes(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, *, x_scale: float = 0.2, y_scale: float = 0.5,
                y_eps: float = 0.5, x_eps: float | int = 1, spk_eps: float = 0.25, 
                title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                linewidth=2, x_min = None, x_max = None, x_range: int = 100,
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

    fig_size = ((x_max - x_min) * x_scale, sum(layer_sizes) * y_scale)
    fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                            figsize=fig_size, layout="constrained")
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
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


def plot_traces(simulator: 'SNNSimulator', x_scale: float = 0.2, y_scale: float = 0.8,
                y_eps: float = 0.1, x_eps: int | float = 1, trace_scale: float = 0.8, x_min = None, x_max = None, x_range: int = 100,
                drawstyle: str = 'steps-post',
                title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                savepath: str | Path = None, show: bool = True, **kwargs):
    """
    Plot traces
    """
    assert simulator.record_traces, "Trace recording is not enabled."

    if x_eps < 1 and x_eps > 0:
        x_eps = x_eps * simulator.num_steps
    if x_max is None:
        x_max = simulator.num_steps
    if x_min is None:
        x_min = max(0, x_max - x_range)
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]

    num_layers = simulator.network.num_layers
    layer_sizes = simulator.network.layer_sizes
    fig_size = ((x_max - x_min) * x_scale, sum(layer_sizes) * y_scale)
    fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                            figsize=fig_size, layout="constrained")
    for i, layer_traces in enumerate(reversed(simulator.trace_recorder.values)):
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

    ax.set_xlabel(f"Time ({simulator.dt} s)")
    fig.suptitle(title if title is not None else "Neuron Traces", fontsize=20)
    fig.supylabel("Neuron Index")
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


def plot_membranes(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, spike_values: List[np.ndarray] = None, 
                   thresholds: List[float | np.ndarray] = None, *, 
                   x_scale: float = 0.2, y_scale: float = 3.0, title: str = None, plot_inputs: bool = False, 
                   color: str = "blue", cmap: str = None, cmap_range: tuple = (0, 1), layout: str = "constrained",
                   x_min = None, x_max = None, x_range: int = 100,
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

    if not plot_inputs:
        thresholds = thresholds[1:] if thresholds is not None else None
        spike_times = spike_times[1:] if spike_times is not None else None
        values = values[1:]
        layer_sizes = layer_sizes[1:]
    nrows = max(layer_sizes)
    ncols = len(layer_sizes)
    fig, axs = plt.subplots(figsize=(x_scale*ncols*(x_max - x_min), y_scale*nrows), layout=layout)
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
        plt.savefig(savepath)

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
                 color_scale: str = "linear",
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
    fig, axs = plt.subplots(1, num_layers, figsize=(width, height), squeeze=False, layout="constrained", gridspec_kw={"hspace": 0, "wspace": 0})
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
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_weight_over_time(simulator: 'SNNSimulator' = None, values: List[np.ndarray] = None, *, title: str = None, x_min=None, x_max=None,
                          synapse_layer: int = 0, 
                          savepath=None, show=True, ):
    if simulator is not None:
        assert simulator.record_weights, "Weight recording is not enabled."
    values = simulator.weight_recorder.values if simulator is not None else values
    T = simulator.num_steps if simulator is not None else max([val.shape[2] for val in values])
    x_max = T if x_max is None else x_max
    x_min = 0 if x_min is None else x_min
    L = synapse_layer if synapse_layer < len(values) else 0
    dt = f"{simulator.dt} s" if simulator is not None else "1 unit"

    nrow, ncol = values[L].shape[:2]
    w_mat = values[L]
    wmax = max(np.max(w_mat), 1)
    wmin = min(np.min(w_mat), 0)

    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow), sharex=True, sharey=True, gridspec_kw={"hspace": 0, "wspace": 0},
                            squeeze=False)
    for i in range(nrow):
        for j in range(ncol):
            ax = axs[i, j]
            ax.plot(w_mat[i, j, :])
            ax.set_ylim(wmin, wmax)
            ax.set_xlim(x_min, x_max)
            if i == 0:
                ax.text(0.5, 1.05, f"Post Neuron {j}", transform=ax.transAxes, fontsize=16, ha="center")
            if j == ncol - 1:
                ax.text(1.05, 0.5, f"Pre Neuron {i}", transform=ax.transAxes, fontsize=16, rotation=-90, va="center")
    plt.tight_layout()
    # plt.suptitle(title, y=0.9)
    # Labelling
    fig.text(s=f"Time ({dt})", fontsize=12, x=0.5, y=-0.005, ha='center')
    fig.text(s="Weight Value", fontsize=12, ha='center', x=-0.005, y=0.5, rotation=90, va='center')
    fig.text(s=title if title is not None else "SNN Weight over Time", fontsize=20, x=0.5, y=1.00, ha='center')

    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


def plot_weight_heatmap(simulator: 'SNNSimulator', *, x_scale: float = 0.2, y_scale: float = 0.8,
                        synapse_layer: int = 0, t_min: int = None, t_max: int = None, t_range: int = 100,
                        log_scale: bool = False, cmap: str = "viridis",
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
    fig_size = ((t_max - t_min) * x_scale, num_outputs * num_inputs * y_scale)
    fig, axs = plt.subplots(num_outputs, 1, figsize=fig_size, sharex=True, gridspec_kw={"hspace": 0.0}, squeeze=False)
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
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


def plot_eligibility_traces(simulator: 'SNNSimulator' = None, values: np.ndarray = None, *, 
                            synapse_layer: int = 0, etype: Literal["pre", "post"] = "pre",
                            x_scale: float = 0.2, y_scale: float = 0.8,
                            t_min: int = None, t_max: int = None, t_range: int = 100,
                            cmap: str = "viridis", 
                            savepath: str | Path = None, show: bool = True):
    if simulator is not None:
        if etype == "pre":
            assert simulator.record_eligibility_pre, "Pre-synaptic eligibility trace recording is not enabled."
            etrace = simulator.eligibility_pre_recorder.values[synapse_layer]
        else:
            assert simulator.record_eligibility_post, "Post-synaptic eligibility trace recording is not enabled."
            etrace = simulator.eligibility_post_recorder.values[synapse_layer]
        
        num_inputs = simulator.network.input_size
        num_outputs = simulator.network.output_size
        num_steps = simulator.num_steps
    elif values is not None:
        etrace = values
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


    fig_size = ((t_max - t_min) * x_scale, num_outputs * num_inputs * y_scale)
    fig, axs = plt.subplots(num_outputs, 1, figsize=fig_size, sharex=True, layout="constrained", gridspec_kw={"hspace": 0.0},
                            squeeze=False)

    for i, j in enumerate(reversed(range(num_outputs))):
        ax = axs[i, 0]
        m = ax.imshow(etrace[:, j, t_min:t_max], cmap=cmap, aspect='auto', origin="lower")
        ax.xaxis.set_ticks(np.arange(0, t_max - t_min, 10), labels= np.arange(t_min, t_max, 10))
        ax.set_ylabel(f'Neuron {j}')

    fig.colorbar(m, label='Eligibility Traces', ax=axs)
    axs[-1, 0].set_xlabel("Time steps")
    fig.suptitle(f"Eligibility Traces\nSynapse Layer {synapse_layer}", fontsize=16)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


def plot_intermediate_fitness(simulator: 'SNN_Simulator' = None, values: np.ndarray = None, *, plot_exploration: bool = False,
                              num_steps: int = None, timestamps: np.ndarray = None,
                              x_scale: float = 0.01, y_scale: float = 1.0, x_eps: int = 1,
                              t_min: int = None, t_max: int = None, t_range: int = None, window_size: int = 10,
                              savepath: str | Path = None, show: bool = True):
    if simulator is not None:
        fts = simulator.get_intermediate_fitness(use_portion=False)
        ft = simulator.get_fitness()
        T = simulator.num_steps
        eps_len = simulator.reward_collector.get_episode_lengths()
        eps_timestamp = np.cumsum(eps_len) * simulator.spike_coder.input_delay
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
    runavg = np.convolve(fts, np.ones(window_size) / window_size, mode='same')

    if t_range is None:
        t_range = T
    if t_max is None:
        t_max = T
    if t_min is None:
        t_min = max(0, t_max - t_range)


    fig, axs = plt.subplots(1 + int(plot_exploration), 1, figsize=((t_max - t_min) * x_scale, 10 * y_scale * (int(plot_exploration) + 1)), layout="constrained",
                            squeeze=False)
    ax = axs[0, 0]
    ax.plot(
        ts, fts,
        color="gray", alpha=0.8,
        drawstyle="steps-pre",
        linewidth=1, label="Intermediate Fitness",
        marker="o", markersize=20
    )
    ax.plot(ts, runavg, color="blue", linewidth=2, label=f"Running Average ({window_size})", markersize=10, marker="o")
    ax.legend(loc="lower right", fontsize=12)
    if num_steps is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.set_xlim(t_min - x_eps, t_max + x_eps)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Fitness")
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
    fig.text(0.5, 1.02, f"Average Fitness: {ft:.2f}", ha='center', fontsize=14)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


### Plotting functions for Learning Rule ###

def plot_learning_rule(lrule: 'LearningRule', simulator: 'SNNSimulator' = None, *, 
                       cmap: str = "RdBu", num_mesh: int = 100,
                       wmax: float = 1.0, wmin: float = -1.0, emax: float = 2.0, emin: float = 0.0,
                       rew_list: List[float] = None,
                       x_scale: float = 1.0, y_scale: float = 1.0,
                       savepath: str | Path = None, show: bool = True):

    if simulator is not None:
        assert simulator.record_eligibility_pre or simulator.record_eligibility_post, "Eligibility trace recording is not enabled."
        assert simulator.record_weights, "Weight recording is not enabled."
        wmax = max(wmax, simulator.weight_recorder.values[0].max())
        wmin = min(wmin, simulator.weight_recorder.values[0].min())
        emax = max(emax, simulator.eligibility_pre_recorder.values[0].max())
        emin = min(emin, simulator.eligibility_pre_recorder.values[0].min())
    else:
        wmax = wmax
        wmin = wmin
        emax = emax
        emin = emin

    arule = lrule.ann
    N = num_mesh
    # Eligibility trace range
    etrace_r = np.linspace(emin, emax, N)
    # Weight range
    weight_r = np.linspace(wmin, wmax, N)
    ee, ww = np.meshgrid(etrace_r, weight_r)

    # Reward range
    if rew_list is not None:
        rew_r = np.array(rew_list)
    else:
        rew_r = np.array([-1.0, -0.1, 0.1, 1.0])
    rr = np.full((N, N), rew_r[0])

    # Combine
    dw_rec = np.zeros((N, N, len(rew_r)))

    for i, r in enumerate(rew_r):
        rr = np.full((N, N), r)
        inp = np.concatenate([ww[..., np.newaxis], rr[..., np.newaxis], ee[..., np.newaxis]], axis=2)
        # inp = np.concatenate([rr[..., np.newaxis], ee[..., np.newaxis]], axis=2)
        dw = arule.forward(inp)
        dw_rec[..., i] = dw.reshape(N, N)
    
    # Start plotting
    ncol = int(np.ceil(np.sqrt(len(rew_r))))
    nrow = int(np.ceil(len(rew_r) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 7.5 * x_scale, nrow * 7.5 * y_scale), squeeze=False)

    for i, r in enumerate(rew_r):
        ax = axs.flat[i]
        ax.imshow(dw_rec[..., i], extent=[emin, emax, wmin, wmax], origin='lower', aspect='auto', vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(f"Reward = {r}")
        ax.set_xlabel("Eligibility Trace")
        ax.set_ylabel("Weight")

    fig.colorbar(axs[0, 0].images[0], ax=axs, orientation='vertical', fraction=.1, label='ΔWeight')
    fig.text(0.5, 0.9, f"Learning Rule Response to Inputs:\nf({lrule.input_order}) -> ΔWeight", ha="center", transform=fig.transFigure, fontsize=16)
    
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


### Plotting functions for Evolutionary Results ###


def plot_fitness_generation(file_path: str | Path, *, estimator: str = "mean", errorband: str | tuple = ("pi", 100),
                            linecolor_best: str = "black", linecolor_est: str = "blue", pointcolor: str = "gray",
                            sns_style: str = "whitegrid", sns_palette: str = "muted",
                            x_eps: int = 2, x_scale: float = 0.3, y_scale: float = 1.3, y_size: float = 10,
                            savepath: str | Path = None, show: bool = True):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    res = pd.read_csv(f"{file_path}")
    run_name = Path(file_path).parent.stem

    # Calulate best all-time fitness
    best_fts = res.groupby("gen")["avg_fitness"].max().cummax()
    best_fts.rename("best_fitness", inplace=True)
    res = res.join(best_fts, on="gen", how="left")

    num_gens = res["gen"].max() + 1
    best_fts = res["best_fitness"].max()
    fts_range = res["avg_fitness"].max() - res["avg_fitness"].min()

    fig, ax = plt.subplots(1, 1, figsize=(num_gens * x_scale, y_size))
    # sns.set_theme(palette=sns_palette, style=sns_style)
    # Fitness per individual
    sns.stripplot(data=res, x="gen", y="avg_fitness",  size=5, ax=ax, alpha=0.5, color=pointcolor)
    # Best cumulative fitness
    sns.lineplot(data=res, x="gen", y="best_fitness", color=linecolor_best, linewidth=2, ax=ax, label="Best Fitness")
    # Average fitness for each generation
    sns.lineplot(data=res, x="gen", y="avg_fitness", estimator=estimator, errorbar=errorband, ax=ax, color=linecolor_est, linewidth=2, label=f"{estimator.title()} Fitness")
    ax.set_xlim(0-x_eps, num_gens+x_eps)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    # ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.text(num_gens - 1, best_fts, f"{best_fts:.2f}", ha='right', va="bottom", fontsize=16, transform=ax.transData, color=linecolor_best)
    title_main = f"Fitness Over Generations"
    subtitle = f"({estimator.title()} Fitness +/- {errorband[1]} {errorband[0].upper()})"
    fig.text(0.5, 0.95, title_main, ha='center', fontsize=24)
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=16)
    ax.text(1.00, 1.05, f"Run: {run_name}", ha='right', fontsize=16, transform=ax.transAxes)

    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    # plt.close(fig)


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