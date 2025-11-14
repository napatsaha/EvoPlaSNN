import copy
from pathlib import Path
from typing import List, Literal, Optional
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .base import SpikeGenerator

from .utils import LayerRecorder, MatrixRecorder
from .snn import SNN
from .modulation import Modulator, Reward_Modulator, TD_Error_Modulator
# from lrule import LearningRule
from .spikegen import BinaryClassGenerator
from .decoding import get_decoder_class, get_fitnessor_class, BaseDecoder, BaseFitnessor
from .rewarder import create_rewarder, create_collector, RewarderProtocol, CollectorProtocol
from rl.spike_coding import SpikeCoder
from rl.collector import RewardCollector, TrajectoryCollector
import gymnasium as gym


class SNNSimulator:
    decoder: BaseDecoder | None
    fitnessor: BaseFitnessor | None
    rewarder: RewarderProtocol | None
    collector: CollectorProtocol | None
    def __init__(self, network: SNN, #spike_generator: SpikeGenerator = None, 
                 env: gym.Env = None, spike_coder: SpikeCoder = None, reward_collector: RewardCollector = None,
                 trajectory_collector: TrajectoryCollector = None,
                 update_condition: Literal["on-step", "on-end"] = "on-end",
                 modulation: Literal["reward", "td-error"] = None, modulation_params: dict = {},
                 *, 
                #  params: dict = {},
                #  decoder_type: Literal["final", "rate", "latency"] = "final", decoder_params: dict = {},
                #  fitnessor_type: Literal["accuracy", "reward", "cross-entropy", "mse"] = "accuracy", fitnessor_params: dict = {},
                #  num_steps: int = None, num_episodes: int = None,
                 supervised: bool = True, decay: bool = False,
                 decay_method: Literal["time", "constant"] = "time",
                 decay_rate: float = None, decay_cutoff: Optional[int] = None, 
                 record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True, record_thresholds: bool = False,
                 record_weights: bool = False, record_eligibility_pre: bool = False, record_eligibility_post: bool = False,
                 **kwargs):
        # Duration params
        self.num_steps = 0
        # self.num_steps = num_steps
        # self.num_eps = num_episodes

        self.network = network
        # self.spike_generator = spike_generator
        self._learning_rule = network.learning_rule
        self.env = env
        self.spike_coder = spike_coder
        self.reward_collector = reward_collector
        self.trajectory_collector = trajectory_collector

        # Flags to indicate what to record
        self.record_membrane = record_membrane
        self.record_spikes = record_spikes
        self.record_traces = record_traces
        self.record_thresholds = record_thresholds
        self.record_weights = record_weights
        self.record_eligibility_pre = record_eligibility_pre if self.network.use_etrace_pre else False
        self.record_eligibility_post = record_eligibility_post if self.network.use_etrace_post else False

        # Initialize recorders
        self.mem_recorder = LayerRecorder(network.layer_sizes_active) if self.record_membrane else None
        self.spike_recorder = LayerRecorder(network.layer_sizes, dtype=np.int8) if self.record_spikes else None
        self.trace_recorder = LayerRecorder(network.layer_sizes, dtype=np.float32) if self.record_traces else None
        self.threshold_recorder = LayerRecorder(network.layer_sizes, dtype=np.float32) if self.record_thresholds else None
        self.weight_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_weights else None
        self.eligibility_pre_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_eligibility_pre else None
        self.eligibility_post_recorder = MatrixRecorder([synapse.weights.shape for synapse in network.synapse_layers]) if self.record_eligibility_post else None

        # # Learning related attributes
        # self._supervised = supervised
        # # Enforce unsupervised learning if the learning rule update condition is on-spike
        # if hasattr(self.learning_rule, "condition") and self.learning_rule.condition == "on-spike":
        #     self._supervised = False
        self._soft_reset = self.network.use_soft_reset()
        self.update_condition = update_condition

        # Neuro-modulation
        self._modulation = modulation if modulation is not None else False
        if self._modulation == "reward":
            self.modulator = Reward_Modulator(**modulation_params)
        elif self._modulation == "td-error":
            self.modulator = TD_Error_Modulator(**modulation_params)
        else:
            self.modulator = None

        # Deal with decaying exploration over course of simulation
        self._use_decay = decay
        self._decay = decay
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        self.decay_cutoff = decay_cutoff
        self._explore = self._should_explore(0)
        self.decay_init_value = self.network.get_exploration_rate(simplify=True)
        # if self._decay and self.reward_collector is not None:
        #     self.reward_collector.cutoff_timestep = decay_cutoff ### *** PROBLEMATIC

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
        if self.record_thresholds:
            self.threshold_recorder.reset()
        if self.record_weights:
            self.weight_recorder.reset()
        if self.record_eligibility_pre:
            self.eligibility_pre_recorder.reset()
        if self.record_eligibility_post:
            self.eligibility_post_recorder.reset()
        # self.reward_manager.reset()
        # if hasattr(self, 'decoder') and self.decoder is not None:
        #     self.decoder.reset()
        # if hasattr(self, 'fitnessor') and self.fitnessor is not None:
        #     self.fitnessor.reset()
        # if hasattr(self, 'rewarder') and self.rewarder is not None:
        #     self.rewarder.reset()
        # if hasattr(self, 'collector') and self.collector is not None:
        #     self.collector.reset()

        # # Reset spike generator
        # self.spike_generator.reset()
        self.env.reset()
        self.spike_coder.reset()
        if self.reward_collector is not None:
            self.reward_collector.reset()
        if self.trajectory_collector is not None:
            self.trajectory_collector.reset()

        # Reset network
        self.network.reset()

        # Reset decay/exploration
        self._explore = self._should_explore(0)

    def soft_reset(self, deterministic: bool = False):
        """Reset only for evaluation while keeping learned weights."""
        self.num_steps = 0
        self.network.soft_reset()
        self.spike_coder.reset()
        self.env.reset()
        if self.reward_collector is not None:
            self.reward_collector.reset()
        if self.trajectory_collector is not None:
            self.trajectory_collector.reset()

        if deterministic:
            self.network.set_deterministic()
            self._decay = False
            self._explore = False

    def run(self, num_steps: int = None, num_eps: int = None):
        t_start = self.num_steps
        if num_steps is None and num_eps is None:
            raise ValueError("Either num_steps or num_eps must be specified.")
        if num_steps is None and num_eps is not None:
            max_steps_per_eps = getattr(self.env, "max_steps", None)
            if max_steps_per_eps is None:
                raise ValueError("Environment does not have max_steps attribute. num_steps must be specified.")
            num_steps = num_eps * max_steps_per_eps * self.spike_coder.input_delay
        self._setup_run(num_steps)
        # _new_sample = True
        state, info = self.env.reset()
        starting_state = self.env.get_agent_state()
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
            if self.spike_coder.ready and action is not None:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_done = terminated or truncated

                # Optional: Record step data
                if self.trajectory_collector is not None:
                    self.trajectory_collector.collect(
                        state=self.env.get_agent_state() if hasattr(self.env, "get_agent_state") else None,
                        observation=state,
                        action=action,
                        reward=reward,
                        done=episode_done,
                        info=info)
                
                # Updates at every environment step
                if self.update_condition == "on-step":
                    signal = self.modulator.signal(locals=locals()) if self._modulation else reward
                    self.network.update_synapses(reward=signal)
                # Update network at the end of each episode
                elif self.update_condition == "on-end":
                    if episode_done:
                        signal = self.modulator.signal(locals=locals()) if self._modulation else reward
                        self.network.update_synapses(reward=reward)

                # Perform episode reset or proceed to next env step
                if episode_done:
                    if self.reward_collector is not None:
                        self.reward_collector.collect(
                            t=t,
                            episode=episode_count,
                            reward=reward, 
                            episode_length=info.get('step_count', None),
                            starting_state=starting_state,
                            exploration=self.network.get_exploration_rate(simplify=True),
                            terminated=terminated,
                            truncated=truncated)
                    self.env.reset()
                    state, info = self.env.reset()
                    starting_state = self.env.get_agent_state()
                    episode_count += 1
                else:
                    state = next_state
            else:
                episode_done = False
                reward = None



            # Update softmax temperature / exploration rate
            if episode_done and self._explore:
                self._update_exploration_rate(t, episode_count)
                # # Update softmax temp at end of episode
                # if self._explore:
                #     if self.decay_method == "time":
                #         new_rate = self.decay_init_value * np.exp(-t * self.decay_rate / self.decay_cutoff)
                #     elif self.decay_method == "constant":
                #         new_rate = self.network.get_exploration_rate(simplify=True) * self.decay_rate
                #     self.network.set_exploration_rate(new_rate)
                # else:
                #     # Set simulator to deterministic
                #     self.network.set_deterministic()
                #     self._decay = False


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

            # Record thresholds
            if self.record_thresholds:
                for i, thresholds in enumerate(self.network.thresholds):
                    self.threshold_recorder.record(i, t, thresholds)

            # Record weights
            if self.record_weights:
                for i, weights in enumerate(self.network.weights):
                    self.weight_recorder.record(i, t, weights)

            # Record eligibility traces
            if self.record_eligibility_pre:
                for i, etraces in enumerate(self.network.eligibility_traces_pre):
                    self.eligibility_pre_recorder.record(i, t, etraces)
            if self.record_eligibility_post:
                for i, etraces in enumerate(self.network.eligibility_traces_post):
                    self.eligibility_post_recorder.record(i, t, etraces)
            # # Check for start of new sample
            # _new_sample = self.spike_generator.is_final()

            # Refresh neuron and synaptic states
            if self._soft_reset and episode_done:
                self.network.soft_reset()
                # self.spike_coder.reset()

            if num_eps is not None and episode_count >= num_eps:
                self.num_steps = t
                break

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
        if self.reward_collector is None:
            Warning("Reward collector is not set. Fitness cannot be calculated.")
            return None
        return self.reward_collector.get_fitness(cutoff=self.decay_cutoff)

    def get_intermediate_fitness(self, use_cutoff: bool = False) -> List[float] | None:
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
        if self.reward_collector is None:
            Warning("Reward collector is not set. Intermediate fitness cannot be calculated.")
            return None
        return self.reward_collector.get_rewards(cutoff=self.decay_cutoff if use_cutoff else None)

    def get_episode_timestamps(self, use_cutoff: bool = False) -> np.ndarray[int] | None:
        if self.reward_collector is None:
            Warning("Reward collector is not set. Episode timestamps cannot be calculated.")
            return None
        eps_timestamp = self.reward_collector.get_timestamps(cutoff=self.decay_cutoff if use_cutoff else None)
        return eps_timestamp

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
        if self.record_thresholds:
            self.threshold_recorder.setup(num_steps)
        if self.record_weights:
            self.weight_recorder.setup(num_steps)
        if self.record_eligibility_pre:
            self.eligibility_pre_recorder.setup(num_steps)
        if self.record_eligibility_post:
            self.eligibility_post_recorder.setup(num_steps)

    def _should_explore(self, t: int) -> bool:
        """
        Determine if exploration should continue based on decay_cutoff.
        """
        if self.decay_cutoff is None:
            return True  # Continuous decay
        if self.decay_cutoff == 0:
            return False  # Cease exploration immediately
        return t < self.decay_cutoff  # Explore until cutoff

    def _update_exploration_rate(self, t: int, episode_count: int):
        """
        Update the exploration rate based on the decay method and cutoff.
        """
        if not self._decay:
            return

        if self._should_explore(t if self.decay_method == "time" else episode_count):
            if self.decay_method == "time":
                # Decay based on timestep
                new_rate = self.decay_init_value * np.exp(-t * self.decay_rate / (self.decay_cutoff or self.num_steps))
            elif self.decay_method == "constant":
                # Decay by multiplying with a constant
                new_rate = self.network.get_exploration_rate(simplify=True) * self.decay_rate
            self.network.set_exploration_rate(new_rate)
        else:
            # Cease exploration and make deterministic
            self.network.set_deterministic()
            self._decay = False
            self._explore = False

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