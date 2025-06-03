from typing import Tuple, override
from pathlib import Path
import numpy as np
from .base import Solver


class EvolutionStrategy(Solver):
    def __init__(self, ndim, popsize, sigma=0.1, minimise=True):
        super().__init__(popsize)
        self.ndim = ndim
        # self.popsize = popsize
        self.minimise = minimise
        self.sigma = sigma
        self.reset(ndim, popsize)

    def reset(self, ndim, popsize):
        self.current_mu = np.zeros(ndim)
        self.fitnesses = np.zeros(popsize)
        self.best_fitness = None
        self.best_solution = None
        
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
    def ask(self):
        """Generate a population of solutions."""
        solutions = np.random.normal(self.current_mu, self.sigma, (self.popsize, self.ndim))
        self.solutions = solutions
        return solutions
    
    @override
    def tell(self, fitness):
        best_idx = np.argmin(fitness) if self.minimise else np.argmax(fitness)
        # Update self
        self.fitnesses = fitness
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.solutions[best_idx]
        self.current_mu = self.best_solution

        # best_fitness = fitness[best_idx]
        # best_solution = self.solutions[best_idx]
        # self.current_mu = best_solution
        # return best_solution, best_fitness
    
    @override
    def result(self) -> Tuple[np.ndarray, float]:
        """Return the best solution and its fitness."""
        return self.best_solution, self.best_fitness
