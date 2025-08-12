from pathlib import Path
import logging
import time
from tqdm import tqdm

import numpy as np


from .base import BaseSolver, Evaluator


class EvoManager:
    """
    Main class for managing loop of evolutionary optimisation.
    """
    def __init__(self, solver: BaseSolver, evaluator: Evaluator, *, num_trials: int = 1, results_path: str = None,
                 max_generations: int = None, target_fitness: float = None, tolerance: float = 1e-6, 
                 max_stagnation: int = None,
                 record_classes: bool = False, save_best: int = 1,
                #  update_inputs: bool = True,  # Whether to update input classes for each generation
                 **kwargs):
        self.solver = solver
        self.evaluator = evaluator
        self.minimise = self.solver.minimise

        self.num_trials = num_trials
        self.save_best = save_best
        self.results_path = Path(results_path) if results_path is not None else None
        self._logfile_name = "log_generation.log"
        # self.update_inputs = update_inputs
        self.record_classes = record_classes

        # Termination control
        self.max_generations = max(max_generations, 1) # Ensure non-zero and non-negative
        self.target_fitness = target_fitness
        self._check_target_fitness = target_fitness is not None
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
        self._setup_logger()
        self.evaluator.setup_logger(self.results_path)
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

                # Update set of classes
                # self.evaluator.generate_new_classes()
                # if self.record_classes:
                #     self.evaluator.write_classes(self.results_path, gen_count)

                # Set up evaluator before start of evaluation
                self.evaluator.setup_generation(gen_count=gen_count)

                # Evaluate solution
                for i, solution in tqdm(enumerate(solutions), desc="Populations", total=self.solver.popsize, position=1, leave=False):
                    self.evaluator.setup_individual(inv_count=i)
                    fitness_list[i] = self.evaluator.evaluate(solution, num_trials=self.num_trials, gen_count=gen_count, indiv_count=i)

                # Inform solver about fitnesses
                self.solver.tell(fitness_list)

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
        finally:
            pbar.close()
            t1 = time.time()
            dt = t1 - t0


        self.logger.info("Terminating Evolutionary optimisation.")
        self.logger.info(f"Best solution: {best_solution.round(4)}")
        self.logger.info(f"Best fitness: {best_fitness:.3f}")
        self.logger.info(f"Total time taken: {dt // 3600} hours, {(dt % 3600) // 60} minutes, {dt % 60:.2f} seconds")
        self.logger.info(f"Total generations: {gen_count}")
        self.logger.info(f"Results saved to directory: {self.results_path}")

        # Save best solution
        if self.results_path is not None:
            save_path = Path(self.results_path)
            self.solver.save_best(save_path, n=self.save_best, precision=6)

        
