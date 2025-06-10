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
from .decoding import get_decoder_class, BaseDecoder


class SNNSimulator:
    def __init__(self, network: SNN, spike_generator: SpikeGenerator, *, 
                 decoder_type: Literal["final", "rate", "latency"] = "final", decoder_params: dict = {},
                 supervised: bool = True,
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

        # For supervised learning
        self._supervised = supervised
        # Enforce unsupervised learning if the learning rule update condition is on-spike
        if hasattr(self.learning_rule, "condition") and self.learning_rule.condition == "on-spike":
            self._supervised = False

        # self.reward_manager = RewardManager()
        self.decoder: BaseDecoder = get_decoder_class(decoder_type)(buffer_size=spike_generator.pattern_length, 
                                                                    neuron_size=network.output_size, **decoder_params) if self._supervised else None


        self.dt = network.dt

    def reset(self):
        """
        Reset the simulator to its initial state.
        """
        # Reset step count
        self.num_steps = 0

        # Reset recorders
        if self.record_membrane:
            self.mem_recorder.reset()
        if self.record_spikes:
            self.spike_recorder.reset()
        if self.record_traces:
            self.trace_recorder.reset()
        if self.record_weights:
            self.weight_recorder.reset()
        # self.reward_manager.reset()
        if self.decoder is not None:
            self.decoder.reset()

        # Reset spike generator
        self.spike_generator.reset()

        # Reset network
        self.network.reset()

    def run(self, num_steps: int):
        t_start = self.num_steps
        self._setup_run(num_steps)
        for t in range(t_start, self.num_steps):
            # Random input spikes
            spk_in = self.spike_generator.generate()

            # Forward pass
            spk_out = self.network.forward(spk_in)

            if self._supervised and self.spike_generator.active:
               self.decoder.record(spk_out)

            if self._supervised and self.spike_generator.ready:
                label = self.spike_generator.get_label()
                reward = self.decoder.calculate_reward(label)
                # self.reward_collector.append((t, label, reward))
                self.network.update_synapses(reward=reward)

            if not self._supervised:
                # Update synaptic weights every timestep
                self.network.update_synapses(reward=None)


            # TODO: Remove
            # Evaluate reward
            # update_signal = self.spike_generator.return_signal()
            # if update_signal and self._supervised:
            #     # label = self.spike_generator.get_label()
            #     # prediction = np.argmax(spk_out) if spk_out.size > 1 else spk_out
            #     # reward = 1.0 if np.equal(label, prediction) else 0.0
            #     # self.reward_collector.append((t, label, prediction, reward))
            #     reward = self.reward_manager.calculate_reward(self.spike_generator.get_label(), spk_out, t)
            # else:
            #     reward = None

            # TODO: Remove
            # Update synaptic weights
            # if self.learning_rule is not None:
            #     if update_signal:
            #         self.network.update_synapses(reward=reward)
            
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

    # TODO: Add get_fitness
    def get_fitness(self) -> float:
        if self.decoder is None:
            Warning("Decoder is not set. Fitness cannot be calculated.")
            return 0.0
        return self.decoder.get_fitness()

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