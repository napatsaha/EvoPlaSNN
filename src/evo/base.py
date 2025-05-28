from typing import List, Tuple
import numpy as np


class Solver:
    """
    Base solver for all Evolutionary Algorithms.
    """
    def __init__(self, popsize: int):
        self.solutions: List | np.ndarray = []
        self.popsize = popsize
        pass

    def ask(self) -> List:
        """
        Returns and records (internally) a set of solutions.
        """
        raise NotImplementedError("ask method must be implemented by subclasses.")
    
    def tell(self, fitnessses: List):
        """
        Informs current solutions with evaluted fitnesses.
        """
        raise NotImplementedError("tell method must be implemented by subclasses.")
    
    def result(self) -> Tuple[object, float]:
        """
        Returns the best solutions and their fitnesses.
        """
        raise NotImplementedError("result method must be implemented by subclasses.")
    

class Evaluator:
    """
    Base class for evaluation functions.
    """
    def __init__(self):
        pass
        # self.fitnesses: List[float] = []

    def evaluate(self, solution: object) -> float:
        """
        Evaluates a given solution and returns its fitness.
        """
        raise NotImplementedError("evaluate method must be implemented by subclasses.")
    