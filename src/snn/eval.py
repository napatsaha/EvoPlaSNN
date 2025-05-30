import logging
import numpy as np

from evo.base import Evaluator
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
    def __init__(self, num_simulation_steps: int = 100, snn_params: dict = {}, spikegen_params: dict = {}, arule_params: dict = {}):
        super().__init__()
        # self.input_size = input_size
        self.num_simulation_steps = num_simulation_steps

        self.spikegen = BinaryClassGenerator(input_size=snn_params.get("input_size"), **spikegen_params)
        self.arule = ANN_Rule(**arule_params)
        self.snn = SNN(learning_rule=self.arule, **snn_params)

        self.simulator = SNNSimulator(self.snn, self.spikegen, record_weights=False, record_traces=False, record_membrane=False, record_spikes=False)

    def setup_logger(self, log_file: str = None):
        handler = logging.StreamHandler() if log_file is None else logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger("SNN_Evaluator")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            # Prevent adding multiple handlers if logger is already configured
            self.logger.handlers.clear()
        self.logger.addHandler(handler)


    def evaluate(self, genome, num_trials=1):
        # mock_parameters = np.linspace(0, 1, num=25)
        # spikegen = PatternSpikeGenerator(input_size, interval=1)

        self.arule.parameters = genome
        fitnesses = []
        if self.logger:
            self.logger.info(f"\nEvaluating genome: {genome}")

        for i in range(num_trials):
            self.spikegen.reset()
            self.simulator.reset()
            self.snn.reset()

            self.simulator.run(self.num_simulation_steps)

            accuracy = self.simulator.reward_manager.accuracy()
            fitnesses.append(accuracy)
            if self.logger:
                self.logger.info(f"Trial {i+1}/{num_trials}: Fitness = {accuracy:.2f}")
        if self.logger:
            self.logger.info(f"Average Fitness over {num_trials} trials: {np.mean(fitnesses):.2f}\tStd Dev = {np.std(fitnesses):.2f}")

        return np.mean(fitnesses)