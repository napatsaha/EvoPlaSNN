from typing import Tuple, override
from pathlib import Path
import numpy as np

from .base import BaseSolver


class CMA_ES(BaseSolver):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm.

    Based on tutorial paper and subsequent implementation in MATLAB by:
    Hansen, Nikolaus. “The CMA Evolution Strategy: A Tutorial.” arXiv, March 10, 2023. https://doi.org/10.48550/arXiv.1604.00772.

    This version neglects Step-Size Control (`sigma`) in the full implementation, and only applies Rank-$$\mu$$ and Rank-one updates to the covariance matrix.
    Equivalent to Section 3.4, Equation (30) of the tutorial paper.
    """
    def __init__(self, ndim: int = 2, popsize: int = None, minimise: bool = True, *, n_best: int = None, sigma: float = 1.0, seed: int = None, **kwargs):
        super().__init__(ndim, popsize, minimise)
        # Random generation
        self.rng = np.random.default_rng(seed)

        # Dimension and population
        self.n_best = int(n_best) if n_best is not None else self.popsize // 2

        # Sizes and weights
        self.weights = np.log((self.n_best + 1) / 2) - np.log(np.arange(1, self.n_best + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2) # Variance effective selection mass 

        # Distribution parameters
        self.mean = np.zeros(ndim)
        self.cov = np.eye(ndim)
        self.sigma = sigma
        # Evolution path
        self.pc = np.zeros(ndim)

        # Cov Adaptation parameters
        self.c_c = (4 + self.mu_eff / self.ndim) / (self.ndim + 4 + 2 * self.mu_eff / self.ndim)
        self.c_1 = 2 / ((self.ndim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff + 1/self.mu_eff - 2) / ((self.ndim+2)**2 + self.mu_eff))
        
    @override
    def ask(self):
        solutions = self.rng.multivariate_normal(self.mean, self.cov, size=(self.popsize, ))
        self.solutions = solutions
        return self.solutions
    
    @override
    def tell(self, fitnesses):
        # Save best solution and fitness
        super().tell(fitnesses)

        # Select parents
        best_indices = np.argsort(fitnesses)[:self.n_best] if self.minimise else np.argsort(fitnesses)[:-self.n_best-1:-1]
        x_best = self.take_solutions(best_indices, return_array=True)

        # Find new mean
        mean_old = self.mean.copy()
        self.mean = np.sum(x_best * self.weights[:, np.newaxis], axis=0)

        # Update evolution path
        self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * \
            (self.mean - mean_old) / self.sigma
        # Rank 1 cov update
        rank_1_cov = self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]

        # Rank mu cov update
        y_mu = (x_best - mean_old) / self.sigma
        rank_mu_cov = (y_mu.T @ np.diag(self.weights)) @ y_mu

        # Update covariance matrix
        self.cov = (1 - self.c_1 - self.c_mu) * self.cov + \
            self.c_1 * rank_1_cov + \
            self.c_mu * rank_mu_cov

    def __repr__(self):
        return f"CMA-ES(ndim={self.ndim}, popsize={self.popsize}, minimise={self.minimise}, n_best={self.n_best}, sigma={self.sigma}, \
            mu_eff={self.mu_eff}, c_c={self.c_c}, c_1={self.c_1}, c_mu={self.c_mu})"