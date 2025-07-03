import copy
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
from .decoding import get_decoder_class, get_fitnessor_class, BaseDecoder, BaseFitnessor
from .rewarder import create_rewarder, create_collector, RewarderProtocol, CollectorProtocol


class SNNSimulator:
    decoder: BaseDecoder | None
    fitnessor: BaseFitnessor | None
    rewarder: RewarderProtocol | None
    collector: CollectorProtocol | None
    def __init__(self, network: SNN, spike_generator: SpikeGenerator, *, 
                 params: dict = {},
                #  decoder_type: Literal["final", "rate", "latency"] = "final", decoder_params: dict = {},
                #  fitnessor_type: Literal["accuracy", "reward", "cross-entropy", "mse"] = "accuracy", fitnessor_params: dict = {},
                 supervised: bool = True,
                 record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True, record_weights: bool = False):
        self.num_steps = 0
        self.network = network
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

        # Initialize post-processing components
        params = copy.deepcopy(params)
        # Option 0: Decodes spike output after certain timestep and computes reward/fitness based on decoded output
        if "decoder_params" in params or "fitnessor_params" in params:
            self._post_process_type = 0
            if "decoder_params" in params:
                decoder_type = params["decoder_params"].pop("type", "final")
                self.decoder: BaseDecoder = get_decoder_class(decoder_type)(buffer_size=spike_generator.pattern_length, 
                                                                            neuron_size=network.output_size, **params["decoder_params"]) if self._supervised else None
            if "fitnessor_params" in params:
                fitnessor_type = params["fitnessor_params"].pop("type", "accuracy")
                fitnessor_params = params["fitnessor_params"]
                self.fitnessor: BaseFitnessor = get_fitnessor_class(fitnessor_type)(num_classes=network.output_size, **fitnessor_params) if self._supervised else None
        # Option 1: Computes reward at every timestep by comparing spike outputs with target outputs. 
        # Fitness is just aggregated version of either errors or rewards within each example.
        elif "rewarder_params" in params:
            self._post_process_type = 1
            rewarder_type = params["rewarder_params"].pop("type", "simple")
            self.rewarder = create_rewarder(
                rewarder_type,
                num_classes=network.output_size, 
                pattern_length=spike_generator.pattern_length, 
                spacing=spike_generator.spacing, 
                **params["rewarder_params"])
            collector_type = params["collector_params"].pop("type", "simple")
            self.collector = create_collector(
                collector_type,
                buffer_size=spike_generator.length, **params.get("collector_params", {}))
        # Option 2: No supervised learning. No reward function.
        else:
            self._post_process_type = -1
            self._supervised = False
            self.decoder = None
            self.fitnessor = None
            self.rewarder = None
            self.collector = None

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
        if hasattr(self, 'decoder') and self.decoder is not None:
            self.decoder.reset()
        if hasattr(self, 'fitnessor') and self.fitnessor is not None:
            self.fitnessor.reset()
        if hasattr(self, 'rewarder') and self.rewarder is not None:
            self.rewarder.reset()
        if hasattr(self, 'collector') and self.collector is not None:
            self.collector.reset()

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

            # Post-processing
            # If using Decoder-Fitnessor scheme
            if self._post_process_type == 0:
                if self._supervised and self.spike_generator.active:
                    self.decoder.record(spk_out)

                if self._supervised and self.spike_generator.ready:
                    label = self.spike_generator.get_label()
                    output = self.decoder.decode()
                    reward = self.decoder.calculate_reward(label, output)
                    self.fitnessor.record(label, output, reward)
                    # self.reward_collector.append((t, label, reward))
                    self.network.update_synapses(reward=reward)

            # If using Rewarder-Collector scheme
            elif self._post_process_type == 1:
                target = self.rewarder.get_target(self.spike_generator.current_class)
                error, reward = self.rewarder.get_reward(target, spk_out)
                self.collector.record(reward, error)

                if self.spike_generator.ready:
                    self.collector.collate()

                self.network.update_synapses(reward=reward)

            else:
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

    def get_fitness(self) -> float | None:
        if self._post_process_type == 0:
            if self.fitnessor is None:
                Warning("Fitnessor is not set. Fitness cannot be calculated.")
                return None
            return self.fitnessor.calculate_fitness()
        elif self._post_process_type == 1:
            if self.collector is None:
                Warning("Collector is not set. Fitness cannot be calculated.")
                return None
            return self.collector.calculate_fitness()
        else:
            return None

    def is_minimise(self) -> bool | None:
        if self._post_process_type == 0:
            return self.fitnessor.minimise if self.fitnessor is not None else None
        elif self._post_process_type == 1:
            return self.collector.minimise if self.collector is not None else None
        else:
            return None

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