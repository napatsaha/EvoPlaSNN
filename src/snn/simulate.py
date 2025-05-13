from pathlib import Path
from typing import List, Literal
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import LayerRecorder, MatrixRecorder
from .utils import plot_neuron
from .snn import SNN
from .lrule import LearningRule
from .spikegen import BinaryClassGenerator, SpikeGenerator


class SNNSimulator:
    def __init__(self, network: SNN, num_steps: int, spike_generator: SpikeGenerator, *, 
                 record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True, record_weights: bool = False):
        self.network = network
        self.num_steps = num_steps
        self.spike_generator = spike_generator
        self.learning_rule = network.learning_rule

        # Initialize recorders
        self.record_membrane = record_membrane
        self.record_spikes = record_spikes
        self.record_traces = record_traces
        self.record_weights = record_weights
        self.mem_recorder = LayerRecorder(network.layer_sizes_active, num_steps) if self.record_membrane else None
        self.spk_recorder = LayerRecorder(network.layer_sizes, num_steps, dtype=np.int8) if self.record_spikes else None
        self.trace_recorder = LayerRecorder(network.layer_sizes, num_steps, dtype=np.float32) if self.record_traces else None
        self.weight_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers], num_steps) if self.record_weights else None

        self.dt = network.dt

    def run(self):
        for t in range(self.num_steps):
            # Random input spikes
            spk_in = self.spike_generator.generate()
            if isinstance(self.spike_generator, BinaryClassGenerator):
                can_update = self.spike_generator.ready
            else:
                can_update = True

            # Forward pass
            spk_out = self.network.forward(spk_in)

            # Update synaptic weights
            if can_update and self.learning_rule is not None:
                self.network.update_synapses()
            
            # Record membrane potentials
            if self.record_membrane:
                for i, membrane in enumerate(self.network.membranes):
                    self.mem_recorder.record(i, t, membrane)

            # Record spikes
            if self.record_spikes:
                for i, spikes in enumerate(self.network.spikes):
                    self.spk_recorder.record(i, t, spikes)

            # Record traces
            if self.record_traces:
                for i, traces in enumerate(self.network.traces):
                    self.trace_recorder.record(i, t, traces)

            # Record weights
            if self.record_weights:
                for i, weights in enumerate(self.network.weights):
                    self.weight_recorder.record(i, t, weights)

    def get_spike_times(self, start=0, end=None) -> List[List[np.ndarray]] | None:
        if self.spk_recorder is None:
            Warning("Spike recording is not enabled.")
            return None
        else:
            tf_spikes = []
            for layer_spikes in self.spk_recorder.values[start:end]:
                tf_layer = []
                for neuron in range(layer_spikes.shape[0]):
                    tf_neuron = np.where(layer_spikes[neuron, :])[0]
                    tf_layer.append(tf_neuron)
                tf_spikes.append(tf_layer)
            return tf_spikes

    def plot_membranes(self, col_width: float = 10.0, row_height: float = 2.5, title: str = None, plot_inputs: bool = True, 
                       color: str = "blue", cmap: str = None, cmap_range: tuple = (0, 1),
                    savepath: str | Path = None, show: bool = True):
        """
        Plot membrane potentials of all neurons in the network.
        """
        assert self.record_membrane, "Membrane recording is not enabled."
        assert self.record_spikes, "Spike recording is not enabled."

        if cmap is None:
            cm = color
        else:
            cm = mpl.colormaps[cmap]

        thr = self.network.thresholds if plot_inputs else self.network.thresholds[1:]
        spike_times = self.get_spike_times() if plot_inputs else self.get_spike_times(start=1)
        mem_values = self.mem_recorder.values if plot_inputs else self.mem_recorder.values[1:]

        layer_sizes = self.mem_recorder.layer_sizes if plot_inputs else self.mem_recorder.layer_sizes[1:]
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
                plot_neuron(fig, gs[j, i], mem=layer_mem[j, :],
                            threshold=thr[i], tf_post=spike_times[i][j], color=c)
                
        # Labelling
        fig.supxlabel(f"Time ({self.dt} s)", fontsize=16, y=0.07)
        fig.supylabel("Membrane Potential", fontsize=16, ha='center', x=0.1)
        fig.suptitle(title if title is not None else "Membrane Potentials", fontsize=20, y=0.9)
        
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')

        if show:
            plt.show()
        plt.close(fig)


    def plot_spikes(self, x_scale: float = 0.2, y_scale: float = 0.5,
                    y_eps: float = 0.5, x_eps: float = 0.02, spk_eps: float = 0.25, 
                    title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                    linewidth=2,
                    savepath: str | Path = None, show: bool = True, **kwargs):
        """
        Plot spike trains with time on x-axis and neuron index on y-axis.
        """
        assert self.record_spikes, "Spike recording is not enabled."

        x_eps = x_eps * self.num_steps
        if cmap is None:
            cm = color
        else:
            cm = mpl.colormaps[cmap]        

        num_layers = self.network.num_layers
        layer_sizes = self.network.layer_sizes
        fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                               figsize=(self.num_steps * x_scale, sum(layer_sizes) * y_scale), layout="constrained")
        for i, layer_spikes in enumerate(reversed(self.spk_recorder.values)):
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
            ax.set_xlim(0 - x_eps, self.num_steps + x_eps)
            ax.set_yticks(np.arange(n_neurons))
            ax.set_yticklabels(np.arange(n_neurons))
            ax.set_ylabel(f"Layer {i}", rotation=90, ha="center")
            # X Grid
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)
        ax.set_xlabel(f"Time ({self.dt} s)")
        # ax.set_ylabel("Neuron Index")
        fig.suptitle(title if title is not None else "Spike Trains", fontsize=20)
        fig.supylabel("Neuron Index")
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        plt.close(fig)

    def plot_traces(self, x_scale: float = 0.2, y_scale: float = 1.0,
                    y_eps: float = 0.1, x_eps: float = 0.0, trace_scale: float = 0.8, 
                    title: str = None, cmap = None, color: str = "black", cmap_range: tuple = (0, 1),
                    savepath: str | Path = None, show: bool = True, **kwargs):
        """
        Plot traces
        """
        assert self.record_traces, "Trace recording is not enabled."

        x_eps = x_eps * self.num_steps
        if cmap is None:
            cm = color
        else:
            cm = mpl.colormaps[cmap]

        num_layers = self.network.num_layers
        layer_sizes = self.network.layer_sizes
        fig, axs = plt.subplots(num_layers, 1, gridspec_kw={"hspace": 0.0}, sharex=True, height_ratios=layer_sizes[::-1], 
                               figsize=(self.num_steps * x_scale, sum(layer_sizes) * y_scale), layout="constrained")
        for i, layer_traces in enumerate(reversed(self.trace_recorder.values)):
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
            ax.set_xlim(0 - x_eps, self.num_steps + x_eps)
            ax.set_yticks(np.arange(n_neurons))
            ax.set_yticklabels(np.arange(n_neurons))
            ax.set_ylabel(f"Layer {i}", rotation=90, ha="center")
            # X Grid
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.xaxis.grid(visible=True, which="both", color="gray", linewidth=0.5, alpha=0.2)

        ax.set_xlabel(f"Time ({self.dt} s)", fontsize=16)
        # ax.set_ylabel("Neuron Index")
        fig.suptitle(title if title is not None else "Neuron Traces", fontsize=20)
        fig.supylabel("Neuron Index", fontsize=16)
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        plt.close(fig)

    def plot_weights(self, div: int = 5, col_width: float = 6.0, row_height: float = 8.0,
                    title: str = None, cmap: str = "gray",
                    savepath: str | Path = None, show: bool = True):
        num_layers = len(self.network.synapse_layers)

        ts = np.linspace(0, self.num_steps-1, div+1).astype(int)

        fig, axs = plt.subplots(num_layers, div+1, figsize=(col_width*div, row_height), squeeze=False)
        fs = np.prod(fig.get_size_inches())/16
        # mpl.rcParams.update({"font.size": np.prod(fig.get_size_inches())/16})

        cmap = mpl.colormaps[cmap].reversed()

        for l in range(num_layers):
            for i, t in enumerate(ts):
                im = self.weight_recorder.values[l][:, :, t]
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
        plt.show()



if __name__ == "__main__":
    pass
    # Parameters
    # input_size = 4
    # hidden_size = [10, 5]
    # output_size = 2
    # tau_mem = 5e-3
    # tau_trace = 1e-3
    # dt = 1e-3
    # threshold = [1, 5, 2]

    # total_timesteps = 100

    # # Create a spike generator
    # spike_gen = RandomSpikeGenerator(input_size, dist="binomial", p=0.5)

    # # Create a spiking network
    # network = SNN(input_size, hidden_size, output_size, dt=dt, tau_mem=tau_mem, tau_trace=tau_trace, threshold=threshold, reset_mechanism="zero")

    # # Create a simulator
    # simulator = SNNSimulator(network, total_timesteps, spike_gen)
    # simulator.run()

    # # Plot membrane potentials
    # simulator.mem_recorder.plot(threshold, figtitle="Membrane Potentials", savepath="membrane_potentials.png", show=True)