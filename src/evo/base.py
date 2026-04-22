from abc import ABC
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

    # def generate_new_classes(self) -> None:
    #     """
    #     Update set of classes used for spike generation.
    #     Meant to be called at beginning of each generation.
    #     """
    #     pass

    def setup_generation(self, gen_count: int, **kwargs):
        """
        Sets up at the beginning of each generation.  
        To be called outside the class (i.e. by `Manager`), before whole population is to be evaluated.
        """
        pass

    def setup_individual(self, inv_count: int, **kwargs):
        """
        Sets up at the beginning of each individual evaluation.  
        To be called inside the class (within `evaluate()`), before trial evaluation loop is begun.
        """
        pass

    def setup_trial(self, trial_count: int, **kwargs):
        """
        Sets up at the beginning of each trial evaluation.  
        To be called inside the class (within `evaluate()`), at the start of each iteration of the trial loop.
        """
        pass


class BaseSolver(Solver):
    solutions: List['Genome']

    def __init__(self, ndim: int = 2, popsize: int = None, minimise: bool = True):
        self._solutions = []
        self.ndim = ndim
        self.popsize = popsize if popsize is not None else int(4 + np.floor(3 + np.log(self.ndim)))
        self.minimise = minimise
        self.reset()

    def reset(self):
        self.fitnesses = np.zeros(self.popsize)
        self._solutions = []
        self.best_fitness = None
        self.best_solution = None

    def tell(self, fitnesses):
        best_idx = np.argmin(fitnesses) if self.minimise else np.argmax(fitnesses)
        # Update self
        self.fitnesses = fitnesses
        self.best_fitness = fitnesses[best_idx]
        self.best_solution = self.solutions[best_idx]

    def save_best(self, save_dir: str | Path, n: int = 1, precision: int = 6):
        top_indices = np.argsort(self.fitnesses if self.minimise else -self.fitnesses) # Will arrange from lowest to highest fitness
        top_indices = top_indices[:n]
        # if self.minimise:
        #     # First n lowest fitness
        #     top_indices = top_indices[:n]
        # else:
        #     # Last n fitness in descending order
        #     top_indices = top_indices[-n:][::-1]
        print(top_indices)
        top_solutions = self.take_solutions(top_indices, return_array=False)
        for i in range(n):
            sol = top_solutions[i]
            if isinstance(sol, Genome):
                sol = sol.parameters
            i = str(i + 1).zfill(2)  # Ensure two-digit index
            np.savetxt(Path(save_dir) / f"best_rule_{i}.txt", sol, fmt=f'%.{precision}f')

    @override
    def result(self) -> Tuple['Genome', float]:
        """Return the best solution and its fitness."""
        return self.best_solution, self.best_fitness
    
    def take_solutions(self, indices: int | List | np.ndarray, return_array: bool = False):
        """
        A safe method for bulk indexing Genome objects within `solutions` list.

        If `return_array=True`, returns a concatenated 2D array of Genome parameters.
        If `return_arrray=False`, returns a list of Genome objects.
        """
        if isinstance(indices, int):
            indices = [indices]
        if return_array: # Return a 2D array of concatenated solutions
            return np.c_[[sol.parameters for i, sol in enumerate(self.solutions) if i in indices]]
        else:
            return [sol for i, sol in enumerate(self.solutions) if i in indices]

    @property
    def solutions(self):
        return self._solutions
    @solutions.setter
    def solutions(self, values: List):
        if len(values) == 0:
            self._solutions = []
        elif isinstance(values, List):
            self._solutions = []
            for val in values:
                if isinstance(val, Genome):
                    self._solutions.append(val)
                else:
                    try:
                        sol = Genome(val)
                        self._solutions.append(sol)
                    except:
                        raise ValueError("Cannot convert solutions into Genome")
        elif isinstance(values, np.ndarray):
            self._solutions = []
            for ind in range(self.popsize):
                sol = values[ind]
                try:
                    self._solutions.append(Genome(sol))
                except:
                    raise ValueError("Cannot convert solutions into Genome")



class Genome(ABC):
    """
    Base class to allow for genetic-related operations in evolutionary Solver.
    """
    def __init__(self, parameters = None, **kwargs):
        super().__init__()
        self._parameters = parameters

    def mutate(self) -> 'Genome':
        """
        Create a modified copy of itself
        """
        pass

    @property
    def parameters(self) -> np.ndarray:
        """
        Returns a 1D genetic blueprint of the genome
        """
        return self._parameters

    @property
    def size(self) -> int:
        """
        Returns the number of parameters that exists in the genome
        """
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"Genome({self.parameters})"