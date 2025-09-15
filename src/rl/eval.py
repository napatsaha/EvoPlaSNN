import logging
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import numpy as np

from evo.base import Evaluator
from snn import SNN, SNNSimulator
# from snn.spikegen import BinaryClassGenerator
import snn.spikegen as spkgen
# from snn.spikegen import create_spikegen, create_poisson_class_timing, create_binary_class_timing
# import snn.spikegen

from lrule import ANN_Rule, LearningRule
from rl import ENV_DICT, StateCoder, RewardCollector
    

class RL_Evaluator(Evaluator):
    def __init__(self, 
                 params: Dict = {},
                 record_info: bool = False,
                 log_level: int = 2,
                 learning_rule: LearningRule = None,
                 ):
        super().__init__()
        # self.input_size = input_size
        params = copy.deepcopy(params)
        self.num_simulation_steps = params["num_sim_steps"]
        self.results_path: Path = None
        self._log_info = int(log_level)
        self.record_info = bool(record_info)

        # Generation and other trackers
        self.gen_count = 0
        self.inv_count = 0
        self.trial_count = 0

        # Initialise environment and spike coder
        env_name = params.get("env_params", {}).pop("name", None)
        if env_name not in ENV_DICT:
            raise ValueError(f"Environment '{env_name}' not yet supported. "
                             f"Supported environments: {list(ENV_DICT.keys())}")
        self.env = ENV_DICT.get(env_name)(**params["env_params"])
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        self.spike_coder = StateCoder(num_states, num_actions, **params["spike_coder_params"])
        self.arule = ANN_Rule(**params["arule_params"]) if learning_rule is None else learning_rule
        self.snn = SNN(input_size=num_states, output_size=num_actions, learning_rule=self.arule, **params["snn_params"])
        self.reward_collector = RewardCollector(**params["collector_params"])
        # Update min and max fitness from environment
        self.reward_collector.min_fitness = self.env.get_min_reward()
        self.reward_collector.max_fitness = self.env.get_max_reward()
        
        self.simulator = SNNSimulator(self.snn, self.env, self.spike_coder, self.reward_collector,
                                      record_weights=record_info, 
                                      record_traces=record_info,
                                      record_membrane=record_info,
                                      record_spikes=record_info,
                                      record_thresholds=record_info,
                                      record_eligibility_pre=record_info,
                                      record_eligibility_post=record_info,
                                      **params.get("simulator_params", {})
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
        return self.reward_collector.minimise

    def get_target_fitness(self) -> float:
        """
        Returns the target fitness for the evaluation.
        This is used to determine when the optimisation should stop.
        """
        return self.reward_collector.max_fitness

    def setup_logger(self, results_path: Path = None):
        self.results_path = results_path
        self._log_info = max(1, self._log_info)
        # Set up logging to record runtime information
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
        if self._log_info >= 2:
            self._fits_trial_file = "fitness_per_trial.csv"
            with open(self.results_path / self._fits_trial_file, "w") as f:
                f.write("gen,indiv,trial,fitness,intermediate\n")
            self._fits_indiv_file = "fitness_per_indiv.csv"
            with open(self.results_path / self._fits_indiv_file, "w") as f:
                f.write("gen,indiv,avg_fitness,std_fitness\n")
            self._genome_file = "genome.csv"
            with open(self.results_path / self._genome_file, "w") as f:
                f.write("gen,indiv,genome\n")


    def evaluate(self, genome: np.ndarray = None, num_trials=1, gen_count: int = None, indiv_count: int = None,
                 return_std: bool = False, return_fitness_list: bool = False) -> Union[float, Tuple[float, float], List[float]]:
        if genome is not None:
            self.arule.parameters = genome

        if self._log_info >= 1:
            t00 = time.time()
            self.logger.info(f"Generation {self.gen_count}, Individual {self.inv_count}")
            self.write_genome(self.gen_count, self.inv_count, genome)

        fitnesses = []
        for i in range(num_trials):
            if self._log_info >= 1:
                self.logger.info(f"Trial {i+1}/{num_trials}: Starting Evaluation...")
            # Set up the individual for evaluation
            self.setup_trial(trial_count=i)
            
            # Reset everything
            self.simulator.reset()

            # Run an evaluation
            t0 = time.time()
            self.simulator.run(self.num_simulation_steps)
            t1 = time.time()

            # Final Fitness
            fitness = self.reward_collector.get_fitness()
            fitnesses.append(fitness)
            if self._log_info >= 1:
                self.logger.info(f"Trial {i+1}/{num_trials}: Finished Evaluation. Time taken: {t1 - t0:.4f} seconds.")
                # Get intermediate fitness across samples
                intermediate_fitness = self.simulator.get_intermediate_fitness(use_portion=True)
                self.write_trial(self.gen_count, self.inv_count, i, fitness, intermediate_fitness)
        
        if return_fitness_list:
            return fitnesses        
        
        avg_fitness = np.mean(fitnesses, where=~np.isnan(fitnesses))
        std_fitness = np.std(fitnesses, where=~np.isnan(fitnesses))
        
        if self._log_info >= 1:
            t10 = time.time()
            self.write_indiv(self.gen_count, self.inv_count, avg_fitness, std_fitness)
            self.logger.info(f"Individual evalution time: {t10 - t00:.4f} seconds")

        if return_std:
            return avg_fitness, std_fitness

        else:
            return avg_fitness

    def setup_generation(self, gen_count: int, num_sets: int = None, record_classes: bool = None, **kwargs):
        self.gen_count = gen_count
        # if self.pattern_type == "poisson_a":
        #     params = self.pattern_params.copy()
        #     if num_sets is not None:
        #         params["num_sets"] = num_sets
        #     self.classes = spkgen.create_poisson_class_timing(
        #         **params
        #     )
        #     record_classes = record_classes if record_classes is not None else self._record_classes
        #     if record_classes:
        #         self.write_classes(self.results_path, gen_count)

    def setup_individual(self, inv_count: int, **kwargs):
        self.inv_count = inv_count
        
    def setup_trial(self, trial_count: int, **kwargs):
        self.trial_count = trial_count
        # # If Poisson Type A, update classes with the current set of classes.
        # if self.pattern_type == "poisson_a":
        #     self.spikegen.update_classes(self.classes[trial_count % len(self.classes)])
        # # If Poisson Type B, generate new set of classes for each trial.
        # elif self.pattern_type == "poisson_b":
        #     params = self.pattern_params.copy()
        #     patterns, labels = spkgen.create_poisson_patterns_and_labels(
        #         **params
        #     )
        #     self.classes = patterns
        #     self.spikegen.update_classes(self.classes, labels=labels)

    def write_classes(self, gen):
        pass
        # with open(self.results_path / "classes.txt", "a") as f:
        #     f.write(f"Generation: {gen}\n")
        #     for i, pairs in enumerate(self.classes):
        #         for j, cls in enumerate(pairs):
        #             f.write(f"Pair: {i:>2}, Class: {j:>2}, {cls.tolist()}\n")

    def write_trial(self, gen: int, indiv: int, trial: int, fitness: float, inter_fitness: List[float], precision: int = 1):
        """
        Records generation number, individual number, trial number, final fitness, and intermediate fitness at the end of each trial.
        """
        if self.results_path is not None and self._log_info >= 2:
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
        if self.results_path is not None and self._log_info >= 2:
            with open(self.results_path / self._fits_indiv_file, "a") as f:
                f.write(f"{gen},{indiv},{avg_fitness:.{precision}f},{std_fitness:.{precision}f}\n")

    def write_genome(self, gen: int, indiv: int, genome: np.ndarray, precision: int = 6):
        """
        Records generation number, individual number, and genome for each individual.
        """
        if self.results_path is not None and self._log_info >= 2:
            with open(self.results_path / self._genome_file, "a") as f:
                genome_str = ','.join([f"{param:.{precision}f}" for param in genome])
                f.write(f"{gen},{indiv},{genome_str}\n")