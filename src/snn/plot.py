from pathlib import Path
import pickle
from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np

# from .simulate import SNNSimulator


def plot_spikes(simulator: 'SNNSimulator', x_scale: float = 0.2, y_scale: float = 0.5,
                y_eps: float = 0.5, x_eps: float | int = 1, spk_eps: float = 0.25, 
                title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                linewidth=2, x_min = None, x_max = None,
                savepath: str | Path = None, show: bool = True, **kwargs):
    """
    Plot spike trains with time on x-axis and neuron index on y-axis.
    """
    assert simulator.record_spikes, "Spike recording is not enabled."

    if x_eps < 1 and x_eps > 0:
        x_eps = x_eps * simulator.num_steps
    if x_max is None:
        x_max = simulator.num_steps
    if x_min is None:
        x_min = max(0, x_max - 100)
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]        

    num_layers = simulator.network.num_layers
    layer_sizes = simulator.network.layer_sizes
    fig_size = ((x_max - x_min) * x_scale, sum(layer_sizes) * y_scale)
    fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                            figsize=fig_size, layout="constrained")
    for i, layer_spikes in enumerate(reversed(simulator.spike_recorder.values)):
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
        ax.set_ylabel(f"Layer {i}", rotation=90, ha="center")
        # X Grid
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)
    ax.set_xlabel(f"Time ({simulator.dt} s)")
    # ax.set_ylabel("Neuron Index")
    fig.suptitle(title if title is not None else "Spike Trains", fontsize=20)
    fig.supylabel("Neuron Index")
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_traces(simulator: 'SNNSimulator', x_scale: float = 0.2, y_scale: float = 1.0,
                y_eps: float = 0.1, x_eps: float = 0.0, trace_scale: float = 0.8, x_min = 0, x_max = None,
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
            ax.plot(y, color=c, alpha=1.0, drawstyle='steps-post', **kwargs)

        ax.set_ylim(0 - y_eps, n_neurons + y_eps)
        ax.set_xlim(x_min - x_eps, x_max + x_eps)
        ax.set_yticks(np.arange(n_neurons))
        ax.set_yticklabels(np.arange(n_neurons))
        ax.set_ylabel(f"Layer {i}", rotation=90, ha="center")
        # X Grid
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)

    ax.set_xlabel(f"Time ({simulator.dt} s)", fontsize=16)
    # ax.set_ylabel("Neuron Index")
    fig.suptitle(title if title is not None else "Neuron Traces", fontsize=20)
    fig.supylabel("Neuron Index", fontsize=16)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_weights(simulator: 'SNNSimulator', div: int = 5, col_width: float = 6.0, row_height: float = 8.0,
                title: str = None, cmap: str = "gray",
                savepath: str | Path = None, show: bool = True):
    num_layers = len(simulator.network.synapse_layers)

    ts = np.linspace(0, simulator.num_steps-1, div+1).astype(int)

    fig, axs = plt.subplots(num_layers, div+1, figsize=(col_width*div, row_height), squeeze=False)
    fs = np.prod(fig.get_size_inches())/16
    # mpl.rcParams.update({"font.size": np.prod(fig.get_size_inches())/16})

    cmap = mpl.colormaps[cmap].reversed()

    for l in range(num_layers):
        for i, t in enumerate(ts):
            im = simulator.weight_recorder.values[l][:, :, t]
            ax = axs[l, i]
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"t={t}", fontsize=fs*0.8)
            ax.set(xticks=[], yticks=[])

    fig.supxlabel("Post-synaptic Neuron", y=0.2, fontsize=fs)
    fig.supylabel("Pre-synaptic Neuron", x=0.1, fontsize=fs)
    fig.suptitle(title if title is not None else "Synaptic Weights", fontsize=1.2*fs, y=0.995)
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap), ax=axs,  
                    orientation="horizontal", fraction=0.05, aspect=100, label="Weight")
    fig.subplots_adjust(wspace=0.0, left=0.1, bottom=0.2)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_weight_over_time(simulator: 'SNNSimulator', title="", x_min=0, x_max=None,
                          savepath=None, show=True, ):
    x_max = simulator.num_steps if x_max is None else x_max
    for L in range(len(simulator.weight_recorder.layer_shapes)):
        nrow, ncol = simulator.weight_recorder.layer_shapes[L]
        w_mat = simulator.weight_recorder.values[L]

        fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow), sharex=True, sharey=True, gridspec_kw={"hspace": 0, "wspace": 0})
        for i in range(nrow):
            for j in range(ncol):
                ax = axs[i, j]
                ax.plot(w_mat[i, j, :])

        ax.set_ylim(0, 1)
        ax.set_xlim(x_min, x_max)
        # plt.tight_layout()
        plt.suptitle(title, y=0.9)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_weight_heatmap(simulator: 'SNNSimulator', *, synapse_layer: int = 0, t_min: int = None, t_max: int = None, cmap: str = "viridis",
                        log_scale: bool = False,
                       savepath: str | Path = None, show: bool = True):
    num_outputs = simulator.network.output_size
    if simulator.record_weights is False:
        raise ValueError("Weight recording is not enabled. Please enable it in the simulator configuration.")
    if t_min is None:
        t_min = 0
    if t_max is None:
        t_max = simulator.num_steps

    # Start plotting
    w_mat = simulator.weight_recorder.values[synapse_layer]
    fig, axs = plt.subplots(num_outputs, 1, figsize=(15, 6), sharex=True)
    axs: List[Axes]
    for i in range(num_outputs):
        ax = axs[i]
        m = ax.imshow(w_mat[:, i, t_min:t_max], aspect='auto', cmap=cmap, norm=mpl.colors.LogNorm() if log_scale else None)
        ax.xaxis.set_ticks(np.arange(0, t_max - t_min, 10), labels= np.arange(t_min, t_max, 10))
        ax.set_ylabel(f"Neuron {i}", fontsize=12)
    axs[-1].set_xlabel("Time Steps", fontsize=12)
    fig.colorbar(m, ax=axs, orientation='vertical', label='Weight Value')
    fig.suptitle(f"Weight Heatmap for Synapse Layer {synapse_layer}", fontsize=16)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)



def plot_membranes(simulator: 'SNNSimulator', col_width: float = 10.0, row_height: float = 2.5, title: str = None, plot_inputs: bool = True, 
                    color: str = "blue", cmap: str = None, cmap_range: tuple = (0, 1), x_min = 0, x_max = None,
                savepath: str | Path = None, show: bool = True):
    """
    Plot membrane potentials of all neurons in the network.
    """
    assert simulator.record_membrane, "Membrane recording is not enabled."
    assert simulator.record_spikes, "Spike recording is not enabled."

    if x_max is None:
        x_max = simulator.num_steps
    if cmap is None:
        cm = color
    else:
        cm = mpl.colormaps[cmap]

    thr = simulator.network.thresholds if plot_inputs else simulator.network.thresholds[1:]
    spike_times = simulator.get_spike_times() if plot_inputs else simulator.get_spike_times(start=1)
    mem_values = simulator.mem_recorder.values if plot_inputs else simulator.mem_recorder.values[1:]

    layer_sizes = simulator.mem_recorder.layer_sizes if plot_inputs else simulator.mem_recorder.layer_sizes[1:]
    nrows = max(layer_sizes)
    ncols = len(layer_sizes)
    fig = plt.figure(figsize=(col_width*ncols, row_height*nrows))
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(ncols):
        layer_mem = mem_values[i]
        n_neurons = layer_mem.shape[0]
        for j in range(n_neurons):
            if cmap is None:
                c = cm
            else:
                c = np.interp(j / (n_neurons - 1), (0, 1), cmap_range)
                c = cm(c)
            _plot_neuron(fig, gs[j, i], mem=layer_mem[j, :],
                        threshold=thr[i], tf_post=spike_times[i][j], color=c,
                        x_min=x_min, x_max=x_max)
            
    # Labelling
    fig.supxlabel(f"Time ({simulator.dt} s)", fontsize=16, y=0.07)
    fig.supylabel("Membrane Potential", fontsize=16, ha='center', x=0.1)
    fig.suptitle(title if title is not None else "Membrane Potentials", fontsize=20, y=0.9)
    
    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)


def _plot_spikes_single(ax: Axes, tf_spikes: np.ndarray, x_max: int, label: str = "", x_min: int = 0):

    ax.eventplot(tf_spikes, colors='gray', linelengths=0.5)
    ax.set_ylim(1.0, 1.5)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center')


def _plot_membrane_old(membrane_array, ax: Axes, threshold=None, title=None):
    ax.plot(membrane_array)
    if threshold is not None:
        ax.axhline(threshold, color='gray', linestyle='--')
    if title is not None:
        ax.set_title(title)


def _plot_membrane_single(ax: Axes, mem: np.ndarray, *, threshold: float = None, tf_pre: int = None, tf_post: int = None,
                                title=None, xlabel=None, ylabel=None, x_min: int = None, x_max: int = None, **kwargs):
    T = len(mem)
    x_min = 0 if x_min is None else x_min
    x_max = T if x_max is None else x_max
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
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(ymin - eps, ymax + eps)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)


def _plot_neuron(fig: Figure, gs: gridspec.GridSpec, mem: np.ndarray, *, tf_pre: int = None, tf_post: int = None, threshold: float = None, 
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
    _plot_membrane_single(ax, mem, tf_pre=tf_pre, tf_post=tf_post, threshold=threshold, x_min=x_min, x_max=x_max, **kwargs)

    if tf_pre is not None:
        ax = fig.add_subplot(gs0[plot_idx])
        plot_idx += 1
        _plot_spikes_single(ax, tf_pre, label="Pre", x_min=x_min, x_max=x_max)


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