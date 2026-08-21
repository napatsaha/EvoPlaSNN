import logging
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Sequence, Union, Tuple

import numpy as np
from numpy.typing import ArrayLike

from common.base import Genome, LearningRule, Evaluator
from common.utils import create_learning_rule, make_bc_func
from snn import SNN, SNNSimulator
# from snn.spikegen import BinaryClassGenerator
import snn.spikegen as spkgen
from snn.spike_coding import SpikeCoderEnvWrapper
# from snn.spikegen import create_spikegen, create_poisson_class_timing, create_binary_class_timing
# import snn.spikegen
from lrule import LearningRule
from rl import ENV_DICT, RewardCollector, BaseMaze
from common.base import SpikeCoder

class RL_Evaluator(Evaluator):
    def __init__(self, 
                 params: Dict, *,
                 env_params: Dict = None,
                 record_info: bool = False,
                 log_level: int = 2,
                 record_inter_fitness: bool = True,
                 precision: int = 3,
                #  learning_rule: LearningRule = None,
                 max_steps: int = None,
                 max_episodes: int = None,
                 eval_episodes: int = None,
                 plastic_on_eval: bool = True,
                 measure_behaviour: bool = False,
                 behaviour_params: dict = None,
                 **kwargs
                 ):
        super().__init__()
        # self.input_size = input_size
        params = copy.deepcopy(params)
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.eval_episodes = eval_episodes
        self.plastic_on_eval = plastic_on_eval
        # if self.num_simulation_steps is None:
        #     max_steps_per_eps = params.get("env_params", {}).get("max_steps", 0)
        #     if self.num_episodes is not None:
        #         self.num_simulation_steps = max_steps_per_eps * self.num_episodes
        #     else:
        #         raise ValueError("Either 'simulator_params.num_steps' or ('env_params.max_steps' and 'simulator_params.num_episodes') must be specified in params.")
        self.results_path: Path = None
        self._log_info = int(log_level)
        self.record_info = bool(record_info)
        self.record_inter_fitness = record_inter_fitness
        self.precision = precision if not record_inter_fitness else 1

        # Generation and other trackers
        # self.gen_count = 0
        # self.inv_count = 0
        # self.trial_count = 0

        # Initialise environment and spike coder
        if env_params is None:
            env_params = params.get("env_params", {})
        else:
            env_params = copy.deepcopy(env_params)
        env_name = env_params.pop("name", None)
        if env_name not in ENV_DICT:
            raise ValueError(f"Environment '{env_name}' not yet supported. "
                             f"Supported environments: {list(ENV_DICT.keys())}")
        self.env: BaseMaze = ENV_DICT.get(env_name)(**env_params)
        # num_states = self.env.observation_space.n
        # num_actions = self.env.action_space.n
        self.spike_coder: SpikeCoder = SpikeCoderEnvWrapper(self.env.observation_space, self.env.action_space, 
                                                            **params["spike_coder_params"])
        self.snn = SNN(input_size=self.spike_coder.input_size, output_size=self.spike_coder.output_size, 
                       **params["snn_params"])
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
                                      record_eligibility_stdp=record_info,
                                      record_eligibility_custom=record_info,
                                      **params.get("simulator_params", {})
                                      )
        self.logger = None

        # Dummy Learning Rule for some pre-calculation
        self._lrule_params = params.get("lrule_params", {})
        self._lrule_type = self._lrule_params.pop("type")
        self.dummy_rule = create_learning_rule(self._lrule_type, **self._lrule_params) #if learning_rule is None else learning_rule
        self.measure_behaviour = measure_behaviour
        self.behaviour_params = {} if behaviour_params is None or not self.measure_behaviour else behaviour_params
        if self.measure_behaviour:
            num_grid = self.behaviour_params.get("num_grid", 2)
            normalise = self.behaviour_params.get("normalise", False)
            self.bc_func = make_bc_func(self.dummy_rule.input_order, num_grid, normalise)
            self.bc_dim = num_grid ** len(self.dummy_rule.input_order)
        else:
            self.bc_func = None

    def get_parameter_size(self):
        """
        Returns the number of parameters in the genome required to build an Evolutionary Algorithm.
        """
        return self.dummy_rule.size if self.dummy_rule is not None else None
    
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

    def setup_logger(self, results_path: Path = None, 
                     logfile_name: str = None, fits_trial_file: str = None, 
                     fits_indiv_file: str = None, genome_file: str = None,
                     logger_name: str = "SNN_Evaluator"):
        self.results_path = results_path
        # self._log_info = max(0, self._log_info)
        # Set up logging to record runtime information
        # self._logfile_name = "log_trials.log"
        self._logfile_name = logfile_name
        if self._logfile_name is not None:
            handler = logging.StreamHandler() if results_path is None else logging.FileHandler(results_path / self._logfile_name)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger = logging.getLogger(logger_name if logger_name is not None else __name__)
            self.logger.setLevel(logging.INFO)
            if self.logger.hasHandlers():
                # Prevent adding multiple handlers if logger is already configured
                self.logger.handlers.clear()
            self.logger.addHandler(handler)

        # Setup three csv files: 
        # one for recording fitnesses in each trial
        # one for recording average fitness for each individual
        # one for recording genome of each individual
        if results_path is not None:
            # self._fits_trial_file = "fitness_per_trial.csv"
            self._fits_trial_file = fits_trial_file
            if self._fits_trial_file is not None:
                with open(self.results_path / self._fits_trial_file, "w") as f:
                    if self.record_inter_fitness:
                        f.write("gen,indiv,trial,fitness,intermediate\n")
                    else:
                        f.write("gen,indiv,trial,fitness\n")
            # self._fits_indiv_file = "fitness_per_indiv.csv"
            self._fits_indiv_file = fits_indiv_file
            if self._fits_indiv_file is not None:
                with open(self.results_path / self._fits_indiv_file, "w") as f:
                    f.write("gen,indiv,avg_fitness,std_fitness\n")
            # self._genome_file = "genome.csv"
            self._genome_file = genome_file
            if self._genome_file is not None:
                with open(self.results_path / self._genome_file, "w") as f:
                    f.write("gen,indiv,genome\n")


    def evaluate(self, genome: np.ndarray | LearningRule | Genome = None, num_trials=1, 
                 gen_count: int = None, inv_count: int = None) -> Tuple[List, float, float, ArrayLike]:
        # Prepare output variables:
        fitnesses = []
        avg_fitness = None
        std_fitness = None
        behv = None

        if genome is not None:
            if isinstance(genome, LearningRule):
                rule = genome
            elif isinstance(genome, Genome):
                rule = create_learning_rule(self._lrule_type, parameters=genome.parameters, **self._lrule_params)
            elif isinstance(genome, np.ndarray):
                rule = create_learning_rule(self._lrule_type, parameters=genome, **self._lrule_params)
            else:
                raise ValueError("Parameters passed into evaluate must be either 'LearningRule', 'Genome' or an 'ArrayLike' object.")
            self.snn.learning_rule = rule
            genome: np.ndarray = rule.parameters

        else:
            raise ValueError("A solution must be specified to be evaluated. Supported types: np.ndarray | LearningRule | Genome")

        t00 = time.time()
        if self.logger is not None:
            self.logger.info(f"Generation {gen_count}, Individual {inv_count}") 
        self.write_genome(gen_count, inv_count, genome)

        # Measure behaviour characteristics of the learning rule (currently assumed to be separate from fitness evaluation)
        if self.measure_behaviour:
            behv = self.bc_func(rule=rule)

        for i in range(num_trials):
            # if self._log_info >= 1:
            #     self.logger.info(f"Trial {i+1}/{num_trials}: Starting Evaluation...")
            # Set up the individual for evaluation
            # self.setup_trial(trial_count=i)
            
            # Reset everything
            self.simulator.reset()

            # Training
            t0 = time.time()
            self.simulator.run(num_steps=self.max_steps, num_eps=self.max_episodes)
            t1 = time.time()

            # Evaluation
            self.simulator.soft_reset(deterministic=True)
            self.simulator.run(num_eps=self.eval_episodes, update=self.plastic_on_eval)

            # Final Fitness
            fitness = self.reward_collector.get_fitness()
            fitnesses.append(fitness)

            if self.logger is not None:
                self.logger.info(f"Trial {i+1}/{num_trials}: Time taken: {t1 - t0:.4f} seconds.")
            # Get intermediate fitness across samples
            intermediate_fitness = self.simulator.get_intermediate_fitness(use_cutoff=True) if self.record_inter_fitness else None
            self.write_trial(gen_count, inv_count, i, fitness, intermediate_fitness, precision=self.precision)
        
        # if return_fitness_list:
        #     return fitnesses        
        
        avg_fitness = np.mean(fitnesses, where=~np.isnan(fitnesses))
        std_fitness = np.std(fitnesses, where=~np.isnan(fitnesses))
        

        t10 = time.time()
        if self.logger is not None:
            self.logger.info(f"Individual evalution time: {t10 - t00:.4f} seconds")
        self.write_indiv(gen_count, inv_count, avg_fitness, std_fitness)

        # if return_std:
        #     return avg_fitness, std_fitness
        # else:
        #     return avg_fitness

        return fitnesses, avg_fitness, std_fitness, behv

    # def setup_generation(self, gen_count: int, num_sets: int = None, record_classes: bool = None, **kwargs):
    #     self.gen_count = gen_count
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

    # def setup_individual(self, inv_count: int, **kwargs):
    #     self.inv_count = inv_count
        
    # def setup_trial(self, trial_count: int, **kwargs):
    #     self.trial_count = trial_count
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

    def write_trial(self, gen: int, indiv: int, trial: int, fitness: float, inter_fitness: List[float] = None, precision: int = 1):
        """
        Records generation number, individual number, trial number, final fitness, and intermediate fitness at the end of each trial.
        """
        if self.results_path is not None and self._fits_trial_file is not None:
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
        if self.results_path is not None and self._fits_indiv_file is not None:
            with open(self.results_path / self._fits_indiv_file, "a") as f:
                f.write(f"{gen},{indiv},{avg_fitness:.{precision}f},{std_fitness:.{precision}f}\n")

    def write_genome(self, gen: int, indiv: int, genome: np.ndarray, precision: int = 6):
        """
        Records generation number, individual number, and genome for each individual.
        """
        if self.results_path is not None and self._genome_file is not None:
            with open(self.results_path / self._genome_file, "a") as f:
                genome_str = ','.join([f"{param:.{precision}f}" for param in genome])
                f.write(f"{gen},{indiv},{genome_str}\n")

    def close(self):
        pass

    @property
    def log_level(self) -> int:
        """
        How detailed to record things:  
        Level 0: No files recorded
        Level 1: Genome and Individual average fitness recorded
        Level 2: Trial intermediate fitness recorded
        """
        return self._log_info