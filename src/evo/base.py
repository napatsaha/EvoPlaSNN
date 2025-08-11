from typing import List, Tuple, Protocol, override
from pathlib import Path
import numpy as np


class Solver(Protocol):
    """
    Base solver for all Evolutionary Algorithms.
    """
    def __init__(self, popsize: int):
        pass

    def ask(self) -> List:
        """
        Returns and records (internally) a set of solutions.
        """
        raise NotImplementedError("ask method must be implemented by subclasses.")
    
    def tell(self, fitnesses: List):
        """
        Informs current solutions with evaluted fitnesses.
        """
        raise NotImplementedError("tell method must be implemented by subclasses.")
    
    def result(self) -> Tuple[object, float]:
        """
        Returns the best solutions and their fitnesses.
        """
        raise NotImplementedError("result method must be implemented by subclasses.")
    

class Evaluator(Protocol):
    """
    Base class for evaluation functions.
    """
    def __init__(self):
        pass
        # self.fitnesses: List[float] = []

    def evaluate(self, solution: object, num_trials: int = None) -> float:
        """
        Evaluates a given solution and returns its fitness.
        """
        raise NotImplementedError("evaluate method must be implemented by subclasses.")
    
    def get_parameter_size(self) -> int:
        """
        Returns the size of the parameter space.
        """
        raise NotImplementedError("get_parameter_size method must be implemented by subclasses.")
    
    def setup_logger(self, log_file: str = None):
        """
        Sets up a logger for the evaluator.
        """
        pass

    def generate_new_classes(self) -> None:
        """
        Update set of classes used for spike generation.
        Meant to be called at beginning of each generation.
        """
        pass

class BaseSolver(Solver):
    def __init__(self, ndim: int = 2, popsize: int = None, minimise: bool = True):
        self.solutions: List | np.ndarray = []
        self.ndim = ndim
        self.popsize = popsize if popsize is not None else int(4 + np.floor(3 + np.log(self.ndim)))
        self.minimise = minimise
        self.reset()

    def reset(self):
        self.fitnesses = np.zeros(self.popsize)
        self.solutions = []
        self.best_fitness = None
        self.best_solution = None

    def tell(self, fitnesses):
        best_idx = np.argmin(fitnesses) if self.minimise else np.argmax(fitnesses)
        # Update self
        self.fitnesses = fitnesses
        self.best_fitness = fitnesses[best_idx]
        self.best_solution = self.solutions[best_idx]

    def save_best(self, save_dir: str | Path, n: int = 1, precision: int = 6):
        top_indices = np.argsort(self.fitnesses) # Will arrange from lowest to highest fitness
        if self.minimise:
            # First n lowest fitness
            top_indices = top_indices[:n]
        else:
            # Last n fitness in descending order
            top_indices = top_indices[-n:][::-1]
        top_solutions = self.solutions[top_indices]
        for i in range(n):
            sol = top_solutions[i]
            i = str(i + 1).zfill(2)  # Ensure two-digit index
            np.savetxt(Path(save_dir) / f"best_rule_{i}.txt", sol, fmt=f'%.{precision}f')

    @override
    def result(self) -> Tuple[np.ndarray, float]:
        """Return the best solution and its fitness."""
        return self.best_solution, self.best_fitness