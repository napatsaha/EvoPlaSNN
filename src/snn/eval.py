import logging
import numpy as np

from evo.base import Evaluator
from snn import SNN, SNNSimulator
# from snn.spikegen import BinaryClassGenerator
import snn.spikegen
from lrule import ANN_Rule

    

class SNN_Evaluator(Evaluator):
    def __init__(self, num_simulation_steps: int = 100, snn_params: dict = {}, spikegen_params: dict = {}, arule_params: dict = {}, decoder_params: dict = {},
                 fitnessor_params: dict = {}):
        super().__init__()
        # self.input_size = input_size
        self.num_simulation_steps = num_simulation_steps

        spikegen_cls = getattr(snn.spikegen, spikegen_params.pop("class", "BinaryClassGenerator"))
        self.spikegen = spikegen_cls(input_size=snn_params.get("input_size"), **spikegen_params)
        self.arule = ANN_Rule(**arule_params)
        self.snn = SNN(learning_rule=self.arule, **snn_params)

        decoder_type = decoder_params.pop("type", "final")
        fitnessor_type = fitnessor_params.pop("type", "reward")
        
        self.simulator = SNNSimulator(self.snn, self.spikegen, record_weights=False, record_traces=False, record_membrane=False, record_spikes=False,
                                      decoder_type=decoder_type, decoder_params=decoder_params, 
                                      fitnessor_type=fitnessor_type, fitnessor_params=fitnessor_params)

    def get_parameter_size(self):
        """
        Returns the number of parameters in the genome required to build an Evolutionary Algorithm.
        """
        return self.arule.size

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
        self.arule.parameters = genome
        fitnesses = []
        if self.logger:
            self.logger.info(f"\nEvaluating genome: {genome}")

        for i in range(num_trials):
            # self.spikegen.reset()
            self.simulator.reset()
            # self.snn.reset()

            self.simulator.run(self.num_simulation_steps)

            accuracy = self.simulator.get_fitness()
            fitnesses.append(accuracy)
            if self.logger:
                self.logger.info(f"Trial {i+1}/{num_trials}: Fitness = {accuracy:.2f}")
        if self.logger:
            self.logger.info(f"Average Fitness over {num_trials} trials: {np.mean(fitnesses):.2f}\tStd Dev = {np.std(fitnesses):.2f}")

        return np.mean(fitnesses)