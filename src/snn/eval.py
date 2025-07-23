import logging
import copy
from pathlib import Path
from typing import Dict, Any
import numpy as np

from evo.base import Evaluator
from snn import SNN, SNNSimulator
# from snn.spikegen import BinaryClassGenerator
from snn.spikegen import create_spikegen, create_poisson_class_timing, create_binary_class_timing
# import snn.spikegen

from lrule import ANN_Rule, LearningRule

    

class SNN_Evaluator(Evaluator):
    def __init__(self, 
                 params: Dict = {},
                 record_info: bool = False,
                 log_info: bool = True,
                 update_inputs: bool = True,
                 learning_rule: LearningRule = None,
                #  num_simulation_steps: int = 100, snn_params: dict = {}, spikegen_params: dict = {}, arule_params: dict = {}, 
                #  decoder_params: dict = {}, fitnessor_params: dict = {}
                 ):
        super().__init__()
        # self.input_size = input_size
        params = copy.deepcopy(params)
        self.num_simulation_steps = params["num_sim_steps"]
        self._logfile_name = "trials.log"
        self.results_path: Path = None
        self.record_classes = params["evo_params"]["manager"].get("record_classes", False)
        self._log_info = bool(log_info)
        self.update_inputs = bool(update_inputs)

        # Initialise spike patterns and spike generator
        self.pattern_params = params.pop("pattern_params", None)
        if self.pattern_params is not None:
            input_size = params["snn_params"].get("input_size", None)
            self.pattern_params["input_size"] = input_size
            
            self.pattern_type = self.pattern_params.pop("type", "poisson")
            if self.pattern_type == "poisson":
                self.classes = create_poisson_class_timing(
                    **self.pattern_params
                )
                self.spikegen = create_spikegen(
                    params["spikegen_params"].pop("class", "CustomTimingGenerator"),
                    input_size=input_size,
                    duration=self.pattern_params.get("duration"),
                    timings=self.classes[0],
                    **params["spikegen_params"]
                )
            elif self.pattern_type == "binary":
                self.classes = create_binary_class_timing(
                    **self.pattern_params
                )
                # Create spike generator with binary class timing
                interval = self.pattern_params.get("interval", 1)
                duration = (input_size - 1) * interval + 1
                self.spikegen = create_spikegen(
                    params["spikegen_params"].pop("class", "CustomTimingGenerator"),
                    input_size=input_size,
                    duration=duration,
                    timings=self.classes,
                    **params["spikegen_params"]
                )
            else:
                raise ValueError(f"Unknown pattern type: {self.pattern_type}")
        else:
            self.spikegen = create_spikegen(params["spikegen_params"].pop("class", "BinaryClassGenerator"), 
                                            input_size=params["snn_params"].get("input_size"),
                                            **params["spikegen_params"])
        self.arule = ANN_Rule(**params["arule_params"]) if learning_rule is None else learning_rule
        self.snn = SNN(learning_rule=self.arule, **params["snn_params"])

        # decoder_type = decoder_params.pop("type", "final")
        # fitnessor_type = fitnessor_params.pop("type", "reward")
        
        self.simulator = SNNSimulator(self.snn, self.spikegen, 
                                      record_weights=False if not record_info else True, 
                                      record_traces=False if not record_info else True,
                                      record_membrane=False if not record_info else True,
                                      record_spikes=False if not record_info else True,
                                      params=params
                                    #   decoder_type=decoder_type, decoder_params=decoder_params, 
                                    #   fitnessor_type=fitnessor_type, fitnessor_params=fitnessor_params
                                      )
        self.logger = None

    def get_parameter_size(self):
        """
        Returns the number of parameters in the genome required to build an Evolutionary Algorithm.
        """
        return self.arule.size
    
    def is_minimise(self):
        """
        Returns whether the optimisation is minimisation or maximisation.
        Depends on Fitnessor type.
        """
        return self.simulator.is_minimise()

    def get_target_fitness(self) -> float:
        """
        Returns the target fitness for the evaluation.
        This is used to determine when the optimisation should stop.
        """
        return self.simulator.get_target_fitness()

    def setup_logger(self, results_path: Path = None):
        self.results_path = results_path
        if self._log_info:
            handler = logging.StreamHandler() if results_path is None else logging.FileHandler(results_path / self._logfile_name)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger = logging.getLogger("SNN_Evaluator")
            self.logger.setLevel(logging.INFO)
            if self.logger.hasHandlers():
                # Prevent adding multiple handlers if logger is already configured
                self.logger.handlers.clear()
            self.logger.addHandler(handler)

    def evaluate(self, genome: np.ndarray = None, num_trials=1):
        if genome is not None:
            self.arule.parameters = genome
        fitnesses = []
        if self._log_info:
            self.logger.info(f"\nEvaluating genome: {genome}")

        for i in range(num_trials):
            # Reset everything
            self.simulator.reset()

            # Generate new set of spike pattern sets
            if self.update_inputs:
                self.spikegen.update_classes(self.classes[i])

            self.simulator.run(self.num_simulation_steps)

            fitness = self.simulator.get_fitness()
            fitnesses.append(fitness)
            if self._log_info:
                self.logger.info(f"Trial {i+1}/{num_trials}: Fitness = {fitness:.2f}")
        avg_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        if self._log_info:
            self.logger.info(f"Average Fitness over {num_trials} trials: {avg_fitness:.2f}\tStd Dev = {std_fitness:.2f}")

        return avg_fitness
    
    def update_classes(self):
        if self.update_inputs:
            self.classes = create_poisson_class_timing(
                **self.pattern_params
            )
            # Record generated classes
        if self.record_classes and self.results_path is not None:
            with open(self.results_path / "classes.txt", "a") as f:
                for cls in self.classes:
                    f.write(f"{cls}\n")