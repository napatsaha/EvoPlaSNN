from pathlib import Path
import logging
import time
from typing import List
import yaml
from tqdm import tqdm

import numpy as np

from common.base import Evaluator, Genome, Solver
from common.utils import create_solver
from rl.eval import RL_Evaluator

# from .base import BaseSolver


class EvoManager:
    """
    Main class for managing loop of evolutionary optimisation.
    """
    def __init__(self, config: dict,  #solver: Solver, evaluator: Evaluator, 
                 *, 
                 num_trials: int = 1, 
                 results_path: str = None,
                 multiple_evaluators: bool = False,
                 max_generations: int = None, max_stagnation: int = None,
                 use_target_fitness: bool = None, target_fitness: float = None, tolerance: float = 1e-6, 
                 record_classes: bool = False, save_best: int = 1,
                 log_trial_fts: bool = True, log_trial_info: bool = True, 
                 log_genome: bool = True, log_indiv: bool = True,
                #  update_inputs: bool = True,  # Whether to update input classes for each generation
                 **kwargs):
        self.multiple_evaluators = multiple_evaluators
        

        # Copied from rul_rl.py
        if "arule_params" in config:
            config["lrule_params"] = config.get("lrule_params", {}).update(config["arule_params"])
            del config["arule_params"]

        # Configure SNN Evaluator object
        self.evaluator: Evaluator = None
        self.evaluators: List[Evaluator] = None
        if not self.multiple_evaluators:
            evaluator: Evaluator = RL_Evaluator(
                params=config,
                record_info=False,
                **config["evo_params"]["evaluator"]
            )
            # use_target_fitness = config["evo_params"]["manager"].get("use_target_fitness", False)
            # if use_target_fitness:
            # TODO: Calculate this without Evaluator
            config["evo_params"]["manager"]["target_fitness"] = evaluator.get_target_fitness()  
            # Evaluator info
            self.evaluator = evaluator
            
        else:
            envs_params: dict = config.get("envs_params", {})
            envs_params["envs"] = []
            self.evaluators = []
            
            for file in envs_params.get("files", []):
                with open(file) as f:
                    env_config = yaml.safe_load(f)
                    env_config = env_config.get("env_params", env_config)
                envs_params["envs"].append(env_config)
                evaluator: Evaluator = RL_Evaluator(
                    params=config,
                    env_params=env_config,
                    record_info=False,
                    **config["evo_params"]["evaluator"]
                )
                self.evaluators.append(evaluator)


        # Configure Evolution Solver object
        # if config["lrule_params"]["type"] == "ann":
        #     config["evo_params"]["solver"]["ndim"] = evaluator.get_parameter_size()
        # TODO: Calculate this without Evaluator
        config["evo_params"]["solver"]["minimise"] = evaluator.is_minimise()
        solver = create_solver(config["evo_params"]["solver"], genome_params=config.get("lrule_params").copy())
        if "popsize" not in config["evo_params"]["solver"]:
            config["evo_params"]["solver"]["popsize"] = solver.popsize
        if "ndim" not in config["evo_params"]["solver"]:
            config["evo_params"]["solver"]["ndim"] = solver.ndim
        # Solver info
        self.solver = solver
        self.minimise = self.solver.minimise


        self.config = config





        # Get flag on whether to process a behaviour stage or not
        self.measure_behaviour = evaluator.measure_behaviour

        self.num_trials = num_trials
        self.save_best = save_best
        self.results_path = Path(results_path) if results_path is not None else None
        self._logfile_name = "log_generation.log"
        # self.update_inputs = update_inputs
        self.record_classes = record_classes

        self.log_trial_fts = log_trial_fts
        self.log_trial_info = log_trial_info
        self.log_genome = log_genome
        self.log_indiv = log_indiv

        # Termination control
        self.max_generations = max(max_generations, 1) # Ensure non-zero and non-negative
        self.target_fitness = target_fitness
        self._check_target_fitness = use_target_fitness if use_target_fitness is not None else (target_fitness is not None)
        self.tolerance = tolerance
        self._check_stagnation = True if max_stagnation is not None else False
        self.max_stagnation = max_stagnation

    def _setup_logger(self):
        """
        Set up channels for outputting logging information.
        By default, prints to console.
        """
        self.logger = logging.getLogger("EvoManager")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler() if self.results_path is None else logging.FileHandler(self.results_path / self._logfile_name)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if self.logger.hasHandlers():
            # Prevent adding multiple handlers if logger is already configured
            self.logger.handlers.clear()
        self.logger.addHandler(handler)

    def run(self, logging_freq: int = 1):
        """
        Runs the evolutionary optimisation loop.
        """
        t0 = time.time()
        # Setup logging for each component
        self._setup_logger()
        if self.multiple_evaluators:
            for i, evaluator in enumerate(self.evaluators, start=1):
                evaluator.setup_logger(
                    self.results_path,
                    logfile_name = f"log_trials_env-{i:02d}.log" if self.log_trial_info else None,
                    logger_name = f"SNN_Evaluator_{i:02d}" if self.log_trial_info else None,
                    fits_trial_file = f"fitness_per_trial_env-{i:02d}.csv" if self.log_trial_fts else None, 
                    fits_indiv_file = f"fitness_per_indiv_env-{i:02d}.csv" if self.log_indiv else None, 
                    genome_file = f"genome_env-{i:02d}.csv" if (self.log_genome and (i == 1)) else None
                )
        else:
            self.evaluator.setup_logger(
                    self.results_path,
                    logfile_name = "log_trials.log" if self.log_trial_info else None,
                    fits_trial_file = "fitness_per_trial.csv" if self.log_trial_fts else None, 
                    fits_indiv_file = "fitness_per_indiv.csv" if self.log_indiv else None, 
                    genome_file = "genome.csv" if self.log_genome else None
            )
        self.solver.setup_logger(self.results_path)

        if self.max_generations is None:
            self.max_generations = 1000  # Default maximum generations

        global_best_fitness = np.inf if self.solver.minimise else -np.inf
        stag_count = 0  # Counter for stagnation
        
        try:
            pbar = tqdm(total=self.max_generations, desc="Generations", position=0, leave=True)
            self.logger.info("Starting evolutionary optimisation.")
            gen_count = 0
            while True:
                # Ask for new solutions
                solutions = self.solver.ask()

                fitness_list = np.zeros(self.solver.popsize)
                if self.measure_behaviour:
                    behaviours = []

                # Update set of classes
                # self.evaluator.generate_new_classes()
                # if self.record_classes:
                #     self.evaluator.write_classes(self.results_path, gen_count)

                # Set up evaluator before start of evaluation
                # self.evaluator.setup_generation(gen_count=gen_count)

                # Evaluate solution
                for i, solution in tqdm(enumerate(solutions), desc="Populations", total=self.solver.popsize, position=1, leave=False):
                    # self.evaluator.setup_individual(inv_count=i)
                    if self.multiple_evaluators:
                        fts_per_trial = []
                        for evaluator in self.evaluators:
                            fts_list, _, _, behv = evaluator.evaluate(solution, num_trials=self.num_trials, gen_count=gen_count, inv_count=i)
                            # Since trials is split between multiple evaluators, only raw trial fitness list is used
                            fts_per_trial.extend(fts_list)
                        # Then average out separately across evaluators
                        fitness_list[i] = np.mean(fts_per_trial, where=~np.isnan(fts_per_trial))
                        # Since `behv` is the same for every trial, then we can just use the last one
                        if self.measure_behaviour:
                            behaviours.append(behv)
                    else:
                        fts_list, avg_fts, std_fts, behv = self.evaluator.evaluate(solution, num_trials=self.num_trials, gen_count=gen_count, inv_count=i)
                        fitness_list[i] = avg_fts
                        if self.measure_behaviour:
                            behaviours.append(behv)

                # Inform solver about fitnesses (and optionally behaviour)
                if self.measure_behaviour:
                    self.solver.tell(fitness_list, behaviours, gen_no=gen_count)
                else:
                    self.solver.tell(fitness_list)

                self.solver.write_to_file(gen_no=gen_count)

                # Get best solutions and their fitnesses
                best_solution, best_fitness = self.solver.result()
                if self.solver.minimise:
                    if best_fitness < global_best_fitness:
                        global_best_fitness = best_fitness
                        stag_count = 0
                    else:
                        stag_count += 1
                else:
                    if best_fitness > global_best_fitness:
                        global_best_fitness = best_fitness
                        stag_count = 0
                    else:
                        stag_count += 1

                # Log result
                if gen_count % logging_freq == 0:
                    self.logger.info(f"Generation {gen_count}: Best fitness this generation = {best_fitness:.3f}, All-time best fitness = {global_best_fitness:.3f}, Stagnation count = {stag_count}")

                # Check stopping criteria
                # Reached target fitness
                if self._check_target_fitness and np.abs(best_fitness - self.target_fitness) < self.tolerance:
                    pbar.update(1)
                    self.logger.info(f"TARGET FITNESS: Terminated due to reaching target fitness of {best_fitness:.3f}")
                    break

                # Reached maximum generations
                if gen_count >= (self.max_generations - 1):
                    pbar.update(1)
                    self.logger.info(f"MAXIMUM GENERATIONS: Terminated due to reaching {gen_count} generations.")
                    break
                
                # Stagnation termination
                if self._check_stagnation and stag_count >= self.max_stagnation:
                    pbar.update(1)
                    self.logger.info(f"STAGNATION: Terminated due to having no fitness improvement in the last {stag_count} generations.")
                    break
                
                gen_count += 1
                pbar.update(1)

            # Save best solution
            self.solver.wrapup(n_best=self.save_best)
        finally:
            pbar.close()
            t1 = time.time()
            dt = t1 - t0
            self.solver.close()
            if self.evaluator is not None:
                self.evaluator.close()
            if self.evaluators is not None:
                for evaluator in self.evaluators:
                    evaluator.close()

        if isinstance(best_solution, Genome):
            best_solution = best_solution.parameters.round(4)
        self.logger.info("Terminating Evolutionary optimisation.")
        self.logger.info(f"Best solution: {best_solution}")
        self.logger.info(f"Best fitness: {best_fitness:.3f}")
        self.logger.info(f"Total time taken: {dt // 3600} hours, {(dt % 3600) // 60} minutes, {dt % 60:.2f} seconds")
        self.logger.info(f"Total generations: {gen_count}")
        self.logger.info(f"Results saved to directory: {self.results_path}")

        # if self.results_path is not None:
        #     save_path = Path(self.results_path)
        #     self.solver.save_best(save_path, n=self.save_best, precision=6)

        
