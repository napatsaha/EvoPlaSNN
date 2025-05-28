import numpy as np
import logging

from .base import Solver, Evaluator


class EvoManager:
    """
    Main class for managing loop of evolutionary optimisation.
    """
    def __init__(self, solver: Solver, evaluator: Evaluator, *, 
                 max_generations: int = None, target_fitness: float = None, tolerance: float = 1e-6
                 ):
        self.solver = solver
        self.evaluator = evaluator

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
        handler = logging.StreamHandler()
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
        if self.max_generations is None:
            self.max_generations = 1000  # Default maximum generations
        gen_count = 0
        while True:
            # Ask for new solutions
            solutions = self.solver.ask()

            fitness_list = np.zeros(self.solver.popsize)

            # Evaluate solution
            for i, solution in enumerate(solutions):
                fitness_list[i] = self.evaluator.evaluate(solution)

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
                self.logger.info(f"Best solution: {best_solution}")
                break

            if gen_count >= self.max_generations:
                # Reached maximum generations
                self.logger.info(f"Maximum generations {self.max_generations} reached. Stopping optimisation.")
                self.logger.info(f"Best solution: {best_solution}")
                break
            
            gen_count += 1
