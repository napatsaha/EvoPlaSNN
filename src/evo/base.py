from typing import List, Tuple, Protocol, Union, override
from pathlib import Path
import numpy as np

from common.base import Solver
from common.base import Genome
from common.utils import create_learning_rule
from genome.genome import SimpleGenome


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
    solutions: List

    def __init__(self, ndim: int = 2, popsize: int = None, minimise: bool = True, *,
                 genome_type: str = None, genome_params: dict = None,):
        self._solutions = []
        self.ndim = ndim
        self.popsize = popsize if popsize is not None else int(4 + np.floor(3 + np.log(self.ndim)))
        self.minimise = minimise
        # Genome Info
        self._genome_params = genome_params if genome_params is not None else {}
        if genome_type is not None:
            self._genome_type = genome_type
        else:
            self._genome_type = None
            if self._genome_params is not None and "type" in self._genome_params:
                self._genome_type = self._genome_params.pop("type")
        self._first_gen = True
        self.fitnesses = None
        self.best_fitness = None
        self.best_solution = None
        self.reset()

    def _generate_new_population(self, ):
        self.solutions = []
        for p in range(self.popsize):
            # Generalise solution creation to any type of genome
            if self._genome_type is not None:
                indiv = create_learning_rule(self._genome_type, **self._genome_params)
            else:
                indiv = SimpleGenome(size=self.ndim)
            self.solutions.append(indiv)

    def ask(self) -> List['Genome']:
        if self._first_gen:
            self._first_gen = False
        return self.solutions

    def reset(self):
        self._first_gen = True
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
            i = str(i + 1).zfill(2)  # Ensure two-digit index
            if isinstance(sol, np.ndarray):
                pass
            elif isinstance(sol, Genome):
                sol = sol.parameters
            elif hasattr(sol, "genome"):
                sol = getattr(getattr(sol, "genome"), "parameters")
            else:
                raise TypeError(f"Genome {sol} cannot be written via np.savetxt")
            np.savetxt(Path(save_dir) / f"best_rule_{i}.txt", sol, fmt=f'%.{precision}f')

    @override
    def result(self) -> Tuple['Genome', float]:
        """Return the best solution and its fitness."""
        return self.best_solution, self.best_fitness
    
    def take_solutions(self, indices: int | List | np.ndarray, return_array: bool = False, simplify: bool = True) -> Union['Genome' | List['Genome'] | np.ndarray]:
        """
        A safe method for bulk indexing Genome objects within `solutions` list.

        If `return_array=True`, returns a concatenated 2D array of Genome parameters.
        If `return_arrray=False`, returns a list of Genome objects.
        """
        if isinstance(indices, int) or isinstance(indices, np.int_):
            indices = [indices]
        if return_array: # Return a 2D array of concatenated solutions
            sols = np.c_[[sol.parameters for i, sol in enumerate(self.solutions) if i in indices]]
            if simplify:
                return np.squeeze(sols)
            else:
                return sols
        else:
            sols = [sol for i, sol in enumerate(self.solutions) if i in indices]
            if simplify and len(sols) == 1:
                return sols[0]
            else:
                return sols

    @property
    def solutions(self):
        return self._solutions
    @solutions.setter
    def solutions(self, values: List):
        assert isinstance(values, List), "Solutions must be a list"
        self._solutions = values
        # if len(values) == 0:
        #     self._solutions = []
        # elif isinstance(values, List):
        #     self._solutions = []
        #     for val in values:
        #         if isinstance(val, Genome):
        #             self._solutions.append(val)
        #         else:
        #             try:
        #                 sol = BaseGenome(val)
        #                 self._solutions.append(sol)
        #             except:
        #                 raise ValueError("Cannot convert solutions into Genome")
        # elif isinstance(values, np.ndarray):
        #     self._solutions = []
        #     for ind in range(self.popsize):
        #         sol = values[ind]
        #         try:
        #             self._solutions.append(BaseGenome(sol))
        #         except:
        #             raise ValueError("Cannot convert solutions into Genome")



