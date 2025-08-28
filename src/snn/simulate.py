import copy
from pathlib import Path
from typing import List, Literal
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .base import SpikeGenerator

from .utils import LayerRecorder, MatrixRecorder
from .snn import SNN
# from lrule import LearningRule
from .spikegen import BinaryClassGenerator
from .decoding import get_decoder_class, get_fitnessor_class, BaseDecoder, BaseFitnessor
from .rewarder import create_rewarder, create_collector, RewarderProtocol, CollectorProtocol
from rl.spike_coding import SpikeCoder
from rl.collector import RewardCollector
import gymnasium as gym


class SNNSimulator:
    decoder: BaseDecoder | None
    fitnessor: BaseFitnessor | None
    rewarder: RewarderProtocol | None
    collector: CollectorProtocol | None
    def __init__(self, network: SNN, #spike_generator: SpikeGenerator = None, 
                 env: gym.Env = None, spike_coder: SpikeCoder = None, reward_collector: RewardCollector = None,
                 update_condition: Literal["on-step", "on-end"] = "on-end",
                 *, 
                #  params: dict = {},
                #  decoder_type: Literal["final", "rate", "latency"] = "final", decoder_params: dict = {},
                #  fitnessor_type: Literal["accuracy", "reward", "cross-entropy", "mse"] = "accuracy", fitnessor_params: dict = {},
                 supervised: bool = True, decay: bool = True,
                 decay_rate: float = None, decay_cutoff: int = None, 
                 record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True, record_weights: bool = False,
                 record_eligibility: bool = False):
        self.num_steps = 0
        self.network = network
        # self.spike_generator = spike_generator
        self._learning_rule = network.learning_rule
        self.env = env
        self.spike_coder = spike_coder
        self.reward_collector = reward_collector

        # Initialize recorders
        self.record_membrane = record_membrane
        self.record_spikes = record_spikes
        self.record_traces = record_traces
        self.record_weights = record_weights
        self.record_eligibility = record_eligibility if self.network.use_etrace else False
        self.mem_recorder = LayerRecorder(network.layer_sizes_active) if self.record_membrane else None
        self.spike_recorder = LayerRecorder(network.layer_sizes, dtype=np.int8) if self.record_spikes else None
        self.trace_recorder = LayerRecorder(network.layer_sizes, dtype=np.float32) if self.record_traces else None
        self.weight_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_weights else None
        self.eligibility_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_eligibility else None

        # # Learning related attributes
        # self._supervised = supervised
        # # Enforce unsupervised learning if the learning rule update condition is on-spike
        # if hasattr(self.learning_rule, "condition") and self.learning_rule.condition == "on-spike":
        #     self._supervised = False
        self._soft_reset = self.network.use_soft_reset()
        self.update_condition = update_condition

        # Deal with decaying exploration over course of simulation
        self._use_decay = decay
        self._decay = decay
        self.decay_rate = decay_rate
        self.decay_cutoff = decay_cutoff
        self.decay_init_value = self.network.get_exploration_rate(simplify=True)
        if self._decay:
            self.reward_collector.cutoff_timestep = decay_cutoff

        # # Initialize post-processing components
        # params = copy.deepcopy(params)
        # # Option 0: Decodes spike output after certain timestep and computes reward/fitness based on decoded output
        # if "decoder_params" in params or "fitnessor_params" in params:
        #     self._post_process_type = 0
        #     if "decoder_params" in params:
        #         decoder_type = params["decoder_params"].pop("type", "final")
        #         self.decoder: BaseDecoder = get_decoder_class(decoder_type)(buffer_size=spike_generator.pattern_length, 
        #                                                                     neuron_size=network.output_size, **params["decoder_params"]) if self._supervised else None
        #     if "fitnessor_params" in params:
        #         fitnessor_type = params["fitnessor_params"].pop("type", "accuracy")
        #         fitnessor_params = params["fitnessor_params"]
        #         self.fitnessor: BaseFitnessor = get_fitnessor_class(fitnessor_type)(num_classes=network.output_size, **fitnessor_params) if self._supervised else None
        # # Option 1: Computes reward at every timestep by comparing spike outputs with target outputs. 
        # # Fitness is just aggregated version of either errors or rewards within each example.
        # elif "rewarder_params" in params:
        #     self._post_process_type = 1
        #     rewarder_type = params["rewarder_params"].pop("type", "simple")
        #     self.rewarder = create_rewarder(
        #         rewarder_type,
        #         num_classes=network.output_size, 
        #         pattern_length=spike_generator.pattern_length,
        #         spacing=getattr(spike_generator, "spacing", None),
        #         **params["rewarder_params"])
        #     collector_type = params["collector_params"].pop("type", "simple")
        #     self.collector = create_collector(
        #         collector_type,
        #         buffer_size=spike_generator.pattern_length, **params.get("collector_params", {}))
        # # Option 2: No supervised learning. No reward function.
        # else:
        #     self._post_process_type = -1
        #     self._supervised = False
        #     self.decoder = None
        #     self.fitnessor = None
        #     self.rewarder = None
        #     self.collector = None

        self.dt = network.dt

    def reset(self):
        """
        Reset the simulator to its initial state.
        """
        # Reset step count
        self.num_steps = 0
        if self._use_decay:
            self._decay = True
            self.network.set_stochastic()
            self.network.set_exploration_rate(self.decay_init_value)

        # Reset recorders
        if self.record_membrane:
            self.mem_recorder.reset()
        if self.record_spikes:
            self.spike_recorder.reset()
        if self.record_traces:
            self.trace_recorder.reset()
        if self.record_weights:
            self.weight_recorder.reset()
        if self.record_eligibility:
            self.eligibility_recorder.reset()
        # self.reward_manager.reset()
        if hasattr(self, 'decoder') and self.decoder is not None:
            self.decoder.reset()
        if hasattr(self, 'fitnessor') and self.fitnessor is not None:
            self.fitnessor.reset()
        if hasattr(self, 'rewarder') and self.rewarder is not None:
            self.rewarder.reset()
        if hasattr(self, 'collector') and self.collector is not None:
            self.collector.reset()

        # # Reset spike generator
        # self.spike_generator.reset()
        self.env.reset()
        self.spike_coder.reset()
        if self.reward_collector is not None:
            self.reward_collector.reset()

        # Reset network
        self.network.reset()

    def run(self, num_steps: int):
        t_start = self.num_steps
        self._setup_run(num_steps)
        # _new_sample = True
        state, info = self.env.reset()
        episode_done = False
        episode_count = 0
        for t in range(t_start, self.num_steps):

            # Random input spikes
            # spk_in = self.spike_generator.generate()

            # Encode state as spikes
            spk_in = self.spike_coder.encode(state)

            # Forward pass
            spk_out = self.network.forward(spk_in)

            # Decode spikes into action
            action = self.spike_coder.decode(spk_out)

            # Increment environment step if the spike coder says so
            if self.spike_coder.ready:
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_done = terminated or truncated
                if self.update_condition == "on-step":
                    self.network.update_synapses(reward=reward)
                if episode_done:
                    if self.reward_collector is not None:
                        self.reward_collector.collect(
                            t=t,
                            episode=episode_count,
                            reward=reward, 
                            episode_length=info.get('step_count', None),
                            exploration=self.network.get_exploration_rate(simplify=True))
                    self.env.reset()
                    state, info = self.env.reset()
                    episode_count += 1
            else:
                episode_done = False
                reward = None

            # Update network at the end of each episode
            if episode_done and self.update_condition == "on-end":
                self.network.update_synapses(reward=reward)

            # Update softmax temperature / exploration rate
            if episode_done and self._decay:
                if t < self.decay_cutoff:
                    # Update softmax temp at end of episode
                    new_rate = self.decay_init_value * np.exp(-t * self.decay_rate / self.decay_cutoff)
                    self.network.set_exploration_rate(new_rate)
                else:
                    # Set simulator to deterministic
                    self.network.set_deterministic()
                    self._decay = False


            # # Post-processing
            # # If using Decoder-Fitnessor scheme
            # if self._post_process_type == 0:
            #     if self._supervised and self.spike_generator.active:
            #         self.decoder.record(spk_out)

            #     if self._supervised and self.spike_generator.ready:
            #         label = self.spike_generator.get_label()
            #         output = self.decoder.decode()
            #         pred = self.decoder.predict(output)
            #         reward = self.decoder.calculate_reward(label, pred)
            #         self.fitnessor.record(label=label, output=output, reward=reward, pred=pred) # Prevent changing of arg order
            #         # self.reward_collector.append((t, label, reward))
            #         self.network.update_synapses(reward=reward)

            # # If using Rewarder-Collector scheme
            # elif self._post_process_type == 1:
            #     # Since Rewarder relies on precise timing of spikes and whether the input spikes is active,
            #     # it needs to be updated whenever the spike input pattern changes.
            #     if not self.spike_generator.is_static() and _new_sample:
            #         self.rewarder.update_target_array(self.spike_generator.array, self.spike_generator.get_label())
                    
            #     # target = self.rewarder.get_target(self.spike_generator.get_label())
            #     error, reward = self.rewarder.get_reward(self.spike_generator.get_label(), spk_out)
            #     if self.spike_generator.active:
            #         self.collector.record(reward, error)

            #     if self.spike_generator.ready:
            #         self.collector.collate()

            #     self.network.update_synapses(reward=reward)

            # else:
            #     if not self._supervised:
            #         # Update synaptic weights every timestep
            #         self.network.update_synapses(reward=None)

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

            # Record eligibility traces
            if self.record_eligibility:
                for i, etraces in enumerate(self.network.eligibility_traces):
                    self.eligibility_recorder.record(i, t, etraces)

            # # Check for start of new sample
            # _new_sample = self.spike_generator.is_final()

            # Refresh neuron and synaptic states
            if self._soft_reset and episode_done:
                self.network.soft_reset()

    def get_fitness(self) -> float | None:
        # if self._post_process_type == 0:
        #     if self.fitnessor is None:
        #         Warning("Fitnessor is not set. Fitness cannot be calculated.")
        #         return None
        #     return self.fitnessor.calculate_fitness()
        # elif self._post_process_type == 1:
        #     if self.collector is None:
        #         Warning("Collector is not set. Fitness cannot be calculated.")
        #         return None
        #     return self.collector.calculate_fitness()
        # else:
            # return None
        return self.reward_collector.get_fitness() if self.reward_collector is not None else None

    def get_intermediate_fitness(self, use_portion: bool = False) -> List[float] | None:
        # if self._post_process_type == 0:
        #     if self.fitnessor is None:
        #         Warning("Fitnessor is not set. Intermediate fitness cannot be calculated.")
        #         return None
        #     return self.fitnessor.get_intermediate_fitness(use_portion=use_portion)
        # elif self._post_process_type == 1:
        #     if self.collector is None:
        #         Warning("Collector is not set. Intermediate fitness cannot be calculated.")
        #         return None
        #     return self.collector.get_intermediate_fitness(use_portion=use_portion)
        # else:
            # return None
        return self.reward_collector.get_rewards(use_cutoff=use_portion) if self.reward_collector is not None else None

    def get_target_fitness(self) -> float | None:
        # if self._post_process_type == 0:
        #     if self.fitnessor.minimise:
        #         return 0.0
        #     else:
        #         return 1.0
        # elif self._post_process_type == 1:
        #     if self.collector.fitness_type == "reward":
        #         return float(self.rewarder.get_max_reward())
        #     elif self.collector.fitness_type == "error":
        #         return 0.0
        #     elif self.collector.fitness_type == "mapped":
        #         return 1.0
        # else:
        return None

    def is_minimise(self) -> bool | None:
        # if self._post_process_type == 0:
        #     return self.fitnessor.minimise if self.fitnessor is not None else None
        # elif self._post_process_type == 1:
        #     return self.collector.minimise if self.collector is not None else None
        # else:
        return self.reward_collector.minimise if self.reward_collector is not None else None

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
        if self.record_eligibility:
            self.eligibility_recorder.setup(num_steps)

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

    @property
    def learning_rule(self):
        return self._learning_rule
    @learning_rule.setter
    def learning_rule(self, rule):
        self._learning_rule = rule
        self.network.learning_rule = rule

    def __repr__(self):
        return f"SNNSimulator(network={self.network})"




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