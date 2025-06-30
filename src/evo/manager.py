from pathlib import Path
import numpy as np
import logging

from tqdm import tqdm

from .base import Solver, Evaluator


class EvoManager:
    """
    Main class for managing loop of evolutionary optimisation.
    """
    def __init__(self, solver: Solver, evaluator: Evaluator, *, num_trials: int = 1, log_file: str = None,
                 max_generations: int = None, target_fitness: float = None, tolerance: float = 1e-6, save_best: int = 1,
                 ):
        self.solver = solver
        self.evaluator = evaluator

        self.num_trials = num_trials
        self.save_best = save_best
        self.log_file = log_file

        # Optimsation parameters
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.tolerance = tolerance
        # self.minimise = minimise

    def _setup_logger(self):
        """
        Set up channels for outputting logging information.
        By default, prints to console.
        """
        self.logger = logging.getLogger("EvoManager")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler() if self.log_file is None else logging.FileHandler(self.log_file)
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
        self._setup_logger()
        self.evaluator.setup_logger(Path(self.log_file).with_stem("trials") if self.log_file is not None else None)
        if self.max_generations is None:
            self.max_generations = 1000  # Default maximum generations
        
        self.logger.info("Starting evolutionary optimisation.")
        gen_count = 0
        try:
            pbar = tqdm(total=self.max_generations, desc="Generations", position=0, leave=True)
            while True:
                # Ask for new solutions
                solutions = self.solver.ask()

                fitness_list = np.zeros(self.solver.popsize)

                # Evaluate solution
                for i, solution in tqdm(enumerate(solutions), desc="Populations", total=self.solver.popsize, position=1, leave=False):
                    fitness_list[i] = self.evaluator.evaluate(solution, num_trials=self.num_trials)

                # Inform solver about fitnesses
                self.solver.tell(fitness_list)

                # Get best solutions and their fitnesses
                best_solution, best_fitness = self.solver.result()

                # Log result
                if gen_count % logging_freq == 0:
                    self.logger.info(f"Generation {gen_count}: Best fitness = {best_fitness:.3f}, Best solution = {best_solution.round(2)}")

                # Check stopping criteria
                if np.abs(best_fitness - self.target_fitness) < self.tolerance:
                    # Reached target fitness
                    self.logger.info(f"Target fitness {self.target_fitness} reached at generation {gen_count}.")
                    break

                if gen_count >= self.max_generations:
                    # Reached maximum generations
                    self.logger.info(f"Maximum generations {self.max_generations} reached.")
                    break
                
                gen_count += 1
                pbar.update(1)
        finally:
            pbar.close()


        self.logger.info("Terminating Evolutionary optimisation.")
        self.logger.info(f"Best solution: {best_solution.round(4)}")
        self.logger.info(f"Best fitness: {best_fitness:.3f}")

        # Save best solution
        if self.log_file is not None:
            save_path = Path(self.log_file).parent
            self.solver.save_best(save_path, n=self.save_best, precision=6)

        
