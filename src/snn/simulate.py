from pathlib import Path
from typing import List, Literal
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import LayerRecorder, MatrixRecorder
from .snn import SNN
from lrule import LearningRule
from .spikegen import BinaryClassGenerator, SpikeGenerator
# from  . import plot


class SNNSimulator:
    def __init__(self, network: SNN, spike_generator: SpikeGenerator, *, 
                 record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True, record_weights: bool = False):
        self.network = network
        self.num_steps = 0
        self.spike_generator = spike_generator
        self.learning_rule = network.learning_rule

        # Initialize recorders
        self.record_membrane = record_membrane
        self.record_spikes = record_spikes
        self.record_traces = record_traces
        self.record_weights = record_weights
        self.mem_recorder = LayerRecorder(network.layer_sizes_active) if self.record_membrane else None
        self.spike_recorder = LayerRecorder(network.layer_sizes, dtype=np.int8) if self.record_spikes else None
        self.trace_recorder = LayerRecorder(network.layer_sizes, dtype=np.float32) if self.record_traces else None
        self.weight_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_weights else None

        self.dt = network.dt

    def run(self, num_steps: int):
        t_start = self.num_steps
        self._setup_run(num_steps)
        for t in range(t_start, self.num_steps):
            # Random input spikes
            spk_in = self.spike_generator.generate()
            update_signal = self.spike_generator.return_signal()


            # Forward pass
            spk_out = self.network.forward(spk_in)

            # Update synaptic weights
            if self.learning_rule is not None:
                if update_signal:
                    self.network.update_synapses(update_signal)
            
            # Record membrane potentials
            if self.record_membrane:
                for i, membrane in enumerate(self.network.membranes):
                    self.mem_recorder.record(i, t, membrane)

            # Record spikes
            if self.record_spikes:
                for i, spikes in enumerate(self.network.spikes):
                    self.spike_recorder.record(i, t, spikes)

            # Record traces
            if self.record_traces:
                for i, traces in enumerate(self.network.traces):
                    self.trace_recorder.record(i, t, traces)

            # Record weights
            if self.record_weights:
                for i, weights in enumerate(self.network.weights):
                    self.weight_recorder.record(i, t, weights)

    def get_spike_times(self, start=0, end=None) -> List[List[np.ndarray]] | None:
        if self.spike_recorder is None:
            Warning("Spike recording is not enabled.")
            return None
        else:
            tf_spikes = []
            for layer_spikes in self.spike_recorder.values[start:end]:
                tf_layer = []
                for neuron in range(layer_spikes.shape[0]):
                    tf_neuron = np.where(layer_spikes[neuron, :])[0]
                    tf_layer.append(tf_neuron)
                tf_spikes.append(tf_layer)
            return tf_spikes

    def _setup_run(self, num_steps: int):
        """
        Setup the run by initializing the recorders.
        """
        self.num_steps += num_steps
        if self.record_membrane:
            self.mem_recorder.setup(num_steps)
        if self.record_spikes:
            self.spike_recorder.setup(num_steps)
        if self.record_traces:
            self.trace_recorder.setup(num_steps)
        if self.record_weights:
            self.weight_recorder.setup(num_steps)

    def plot_membranes(self, *args, **kwargs):
        Warning("SNNSimulator.plot_membranes is deprecated. Use plot.plot_membranes(SNN.Simulator) instead.")
        # plot.plot_membranes(self, *args, **kwargs)
        return None

    def plot_weights(self, *args, **kwargs):
        Warning("SNNSimulator.plot_weights is deprecated. Use plot.plot_weights(SNN.Simulator) instead.")
        # plot.plot_weights(self, *args, **kwargs)
        return None

    def plot_spikes(self, *args, **kwargs):
        Warning("SNNSimulator.plot_spikes is deprecated. Use plot.plot_spikes(SNN.Simulator) instead.")
        # plot.plot_spikes(self, *args, **kwargs)
        return None
    
    def plot_traces(self, *args, **kwargs):
        Warning("SNNSimulator.plot_traces is deprecated. Use plot.plot_traces(SNN.Simulator) instead.")
        # plot.plot_traces(self, *args, **kwargs)
        return None

    def __repr__(self):
        return f"SNNSimulator(network={self.network}, spike_generator={self.spike_generator})"




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