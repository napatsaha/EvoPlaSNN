from typing import List, Literal
import numpy as np
from typing import Union
from .utils import LayerRecorder
from .snn import SNN
from .lrule import LearningRule
from .spikegen import SpikeGenerator


class SNNSimulator:
    def __init__(self, network: SNN, num_steps: int, spike_generator: SpikeGenerator, *, record_membrane: bool = True, record_spikes: bool = True, record_traces: bool = True):
        self.network = network
        self.num_steps = num_steps
        self.spike_generator = spike_generator
        self.learning_rule = network.learning_rule
        self.record_membrane = record_membrane
        self.record_spikes = record_spikes
        self.record_traces = record_traces
        self.mem_recorder = LayerRecorder(network.layer_sizes_active, num_steps) if self.record_membrane else None
        self.spk_recorder = LayerRecorder(network.layer_sizes, num_steps, dtype=np.int8) if self.record_spikes else None
        self.trace_recorder = LayerRecorder(network.layer_sizes, num_steps, dtype=np.float32) if self.record_traces else None

    def run(self):
        for t in range(self.num_steps):
            # Random input spikes
            spk_in = self.spike_generator.generate()

            # Forward pass
            spk_out = self.network.forward(spk_in)

            # Update synaptic weights
            if self.learning_rule is not None:
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