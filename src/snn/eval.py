import logging
import copy
import time
from pathlib import Path
from typing import Dict, Any, List
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
        self.results_path: Path = None
        self.record_classes = params["evo_params"]["evaluator"].get("record_classes", False)
        self._log_info = bool(log_info)
        self._update_inputs = bool(params["evo_params"]["evaluator"].get("update_inputs", update_inputs))

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
        # Set up logging to record runtime information
        if self._log_info and self.results_path is not None:
            self._logfile_name = "log_trials.log"
            handler = logging.StreamHandler() if results_path is None else logging.FileHandler(results_path / self._logfile_name)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger = logging.getLogger("SNN_Evaluator")
            self.logger.setLevel(logging.INFO)
            if self.logger.hasHandlers():
                # Prevent adding multiple handlers if logger is already configured
                self.logger.handlers.clear()
            self.logger.addHandler(handler)

            # Setup three csv files: 
            # one for recording fitnesses in each trial
            # one for recording average fitness for each individual
            # one for recording genome of each individual
            self._fits_trial_file = "fitness_per_trial.csv"
            with open(self.results_path / self._fits_trial_file, "w") as f:
                f.write("gen,indiv,trial,fitness,intermediate\n")
            self._fits_indiv_file = "fitness_per_indiv.csv"
            with open(self.results_path / self._fits_indiv_file, "w") as f:
                f.write("gen,indiv,avg_fitness,std_fitness\n")
            self._genome_file = "genome.csv"
            with open(self.results_path / self._genome_file, "w") as f:
                f.write("gen,indiv,genome\n")
        else:
            self.logger.info("No results path provided. Logging to console only.")

    def evaluate(self, genome: np.ndarray = None, num_trials=1, gen_count: int = None, indiv_count: int = None) -> float:
        if genome is not None:
            self.arule.parameters = genome

        if self._log_info:
            t00 = time.time()
            # self.logger.info(f"\nEvaluating genome: {genome}")
            self.logger.info(f"Generation {gen_count}, Individual {indiv_count}")
            self.write_genome(gen_count, indiv_count, genome)

        fitnesses = []
        for i in range(num_trials):
            if self._log_info:
                self.logger.info(f"Trial {i+1}/{num_trials}: Starting Evaluation...")
            # Reset everything
            self.simulator.reset()

            # Generate new set of spike pattern sets
            if self._update_inputs:
                self.spikegen.update_classes(self.classes[i])

            t0 = time.time()
            self.simulator.run(self.num_simulation_steps)
            t1 = time.time()

            # Final Fitness
            fitness = self.simulator.get_fitness()
            # Get intermediate fitness across samples
            intermediate_fitness = self.simulator.get_intermediate_fitness()
            fitnesses.append(fitness)
            if self._log_info:
                self.logger.info(f"Trial {i+1}/{num_trials}: Finished Evaluation. Time taken: {t1 - t0:.4f} seconds.")
                self.write_trial(gen_count, indiv_count, i, fitness, intermediate_fitness)
        avg_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        if self._log_info:
            t10 = time.time()
            self.write_indiv(gen_count, indiv_count, avg_fitness, std_fitness)
            self.logger.info(f"Individual evalution time: {t10 - t00:.4f} seconds")

        return avg_fitness
    
    def update_classes(self):
        if self._update_inputs:
            if self.pattern_type == "poisson":
                self.classes = create_poisson_class_timing(
                    **self.pattern_params
                )
            elif self.pattern_type == "binary":
                self.classes = create_binary_class_timing(
                    **self.pattern_params
                )
            # Record generated classes
        if self.record_classes and self.results_path is not None:
            with open(self.results_path / "classes.txt", "a") as f:
                for cls in self.classes:
                    f.write(f"{cls}\n")

    def write_trial(self, gen: int, indiv: int, trial: int, fitness: float, inter_fitness: List[float], precision: int = 1):
        """
        Records generation number, individual number, trial number, final fitness, and intermediate fitness at the end of each trial.
        """
        if self.results_path is not None:
            with open(self.results_path / self._fits_trial_file, "a") as f:
                if inter_fitness is not None:
                    inter_fitness_str = ','.join([f"{fit:.{precision}f}" for fit in inter_fitness])
                    f.write(f"{gen},{indiv},{trial},{fitness:.{precision}f},{inter_fitness_str}\n")
                else:
                    f.write(f"{gen},{indiv},{trial},{fitness:.{precision}f}\n")

    def write_indiv(self, gen: int, indiv: int, avg_fitness: float, std_fitness: float, precision: int = 3):
        """
        Records generation number, individual number, average fitness, and standard deviation of fitness for each individual.
        """
        if self.results_path is not None:
            with open(self.results_path / self._fits_indiv_file, "a") as f:
                f.write(f"{gen},{indiv},{avg_fitness:.{precision}f},{std_fitness:.{precision}f}\n")

    def write_genome(self, gen: int, indiv: int, genome: np.ndarray, precision: int = 6):
        """
        Records generation number, individual number, and genome for each individual.
        """
        if self.results_path is not None:
            with open(self.results_path / self._genome_file, "a") as f:
                genome_str = ','.join([f"{param:.{precision}f}" for param in genome])
                f.write(f"{gen},{indiv},{genome_str}\n")