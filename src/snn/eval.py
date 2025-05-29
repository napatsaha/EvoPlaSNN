import numpy as np

from ..evo.base import Evaluator
from snn import SNN, SNNSimulator
from snn.spikegen import BinaryClassGenerator
from lrule import ANN_Rule

# class SNN_Evaluator(Evaluator):
#     def __init__(self):
#         super().__init__()

#     def evaluate(self, genome):
#         # mock_parameters = np.linspace(0, 1, num=25)
#         # spikegen = PatternSpikeGenerator(input_size, interval=1)
#         spikegen = BinaryClassGenerator(input_size, interval=interval, spacing=spacing, signal_on_end=True, starting_class=starting_class, p=p, reflect=reflect)

#         arule = ANN_Rule(hidden_size=[4], bias=True, parameters=genome)
#         # srule = STDP_Rule(mu=0.01, lambd=0.1, alpha=0.0, dt=dt)
#         nn = SNN(input_size, hidden_size, output_size, dt=dt, learning_rule=arule, neuron_params={"tau_trace": tau_trace, "tau_mem": tau_mem}, synapse_params=dict(normalise_weights=normalise_weights, normalise_method = normalise_method),
#                 winner_take_all=True)

#         simulator = SNNSimulator(nn, spikegen, record_weights=False, record_traces=False, record_membrane=False)

#         simulator.run(T)

#         fitness = simulator.reward_manager.accuracy()

#         return fitness
    

class SNN_Evaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.spikegen = BinaryClassGenerator(input_size, interval=interval, spacing=spacing, signal_on_end=True, starting_class=starting_class, p=p, reflect=reflect)

        self.arule = ANN_Rule(hidden_size=hidden_size_rule, bias=bias, use_weights=use_weights, use_reward=use_reward, use_trace_pre=use_trace_pre, use_trace_post=use_trace_post)
        # srule = STDP_Rule(mu=0.01, lambd=0.1, alpha=0.0, dt=dt)
        self.snn = SNN(input_size, hidden_size, output_size, dt=dt, learning_rule=self.arule, neuron_params={"tau_trace": tau_trace, "tau_mem": tau_mem}, synapse_params=dict(normalise_weights=normalise_weights, normalise_method = normalise_method),
                winner_take_all=True)

        self.simulator = SNNSimulator(self.snn, self.spikegen, record_weights=False, record_traces=False, record_membrane=False, record_spikes=False)

    def evaluate(self, genome, num_trials=1):
        # mock_parameters = np.linspace(0, 1, num=25)
        # spikegen = PatternSpikeGenerator(input_size, interval=1)

        self.arule.parameters = genome
        fitnesses = []

        for i in range(num_trials):
            self.spikegen.reset()
            self.simulator.reset()
            self.snn.reset()

            self.simulator.run(T)

            # accuracy = np.mean([tup[3] for tup in simulator.reward_collector]).item()
            accuracy = self.simulator.reward_manager.accuracy()
            fitnesses.append(accuracy)

        return np.mean(fitnesses)