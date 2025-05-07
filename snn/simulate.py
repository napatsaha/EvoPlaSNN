from pathlib import Path
from typing import List, Literal
import numpy as np
import matplotlib.pyplot as plt

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

    def get_spike_times(self) -> List[List[np.ndarray]] | None:
        if self.spk_recorder is None:
            Warning("Spike recording is not enabled.")
            return None
        else:
            tf_spikes = []
            for layer_spikes in self.spk_recorder.values:
                tf_layer = []
                for neuron in range(layer_spikes.shape[0]):
                    tf_neuron = np.where(layer_spikes[neuron, :])[0]
                    tf_layer.append(tf_neuron)
                tf_spikes.append(tf_layer)
            return tf_spikes

    def plot_membranes(self, col_width: float = 5.0, row_height: float = 2.5,
             savepath: str | Path = None, show: bool = True):
        
        thr = self.network.thresholds
        spike_times = self.get_spike_times()

        nrows = max(self.mem_recorder.layer_sizes)
        ncols = len(self.mem_recorder.layer_sizes)
        fig = plt.figure(figsize=(col_width*ncols, row_height*nrows))
        gs = fig.add_gridspec(nrows, ncols)

        for i, layer_mem in enumerate(self.mem_recorder.values):
            for j in range(layer_mem.shape[0]):
                plot_neuron(fig, gs[j, i], mem=layer_mem[j, :],
                            threshold=thr[i], tf_post=spike_times[i][j])
                
        # Labelling
        fig.supxlabel("Time (ms)", fontsize=8)
        fig.supylabel("Membrane Potential", fontsize=8)
        fig.suptitle("Membrane Potentials", fontsize=20)
        
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')

        if show:
            plt.show()
        plt.close(fig)



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