# from lrule.cgp import CGP_Graph, CGP_Rule
from common.base import Genome, LearningRule
from common.utils import create_learning_rule
from genome.genome import SimpleGenome
from .base import BaseSolver
from typing import List, Literal

import numpy as np


# class MuPlusLambda(BaseSolver):
#     """
#     Implementing the (\mu + \lambda) specifically tailored for Cartesian Genetic Programming solutions
#     """
#     solutions: List[CGP_Rule]

#     def __init__(self, ndim = 2, popsize = None, minimise = True):
#         super().__init__(ndim, popsize, minimise)

#     def ask(self):
#         pass


class MuPlusLambda(BaseSolver):
    parents: List[Genome]
    # solutions: List[Genome]

    def __init__(self, mu: int, lambd: int, ndim: int = None, minimise: bool = True, *,
                 use_genome_operator: bool = True, 
                 mutation_method: Literal["resample", "perturb"] = "resample", mutation_rate: float = 0.5, mutation_scale: float = 1.0,
                 default_lower_bound: float = 0.0, default_upper_bound: float = 1.0,
                 **kwargs):
        """
        Mu + Lambda Algorithm

        Args:
            mu (int): Number of parents
            lambd (int): Number of offsprings
            minimise (bool, optional): Optimisation direction. Defaults to True.
            mutation_rate (float, optional): Probability of mutating each gene. Defaults to 0.1.
        """
        self.mu = mu
        self.lambd = lambd
        popsize = self.mu + self.lambd

        super().__init__(ndim=ndim, popsize=popsize, minimise=minimise,
                          **kwargs)

        # Gene Operator info
        assert mutation_method in ["resample", "perturb"]
        self.mutation_method = mutation_method
        self.mutation_rate = min(1.0, max(0.0, mutation_rate))
        self.mutation_scale = max(0.0, mutation_scale)

        # Whether or not to use gene object's operators for mutation
        self.use_genome_operator = use_genome_operator
        # If mutating by oneself, need more information 
        if not self.use_genome_operator:
            # Get gene dtypes
            self._gene_dtypes = self.sample_sol.get_gene_dtypes()
            # Get clipping and sampling boundaries
            self.lower_bounds, self.upper_bounds = self.sample_sol.get_bounds()
            # For use only in sampling in mutation_method="resample"
            self._sampling_low = np.where(np.array(self.lower_bounds) == None, default_lower_bound, self.lower_bounds).astype(float)
            self._sampling_high = np.where(np.array(self.upper_bounds) == None, default_upper_bound, self.upper_bounds).astype(float)
            # Replace None with -inf/+inf for use in clipping
            self.lower_bounds = np.where(np.array(self.lower_bounds) == None, -np.inf, self.lower_bounds).astype(float)
            self.upper_bounds = np.where(np.array(self.upper_bounds) == None, np.inf, self.upper_bounds).astype(float)
        else:
            self.lower_bounds = None
            self.upper_bounds = None

        self.parents: List[Genome] = []

        # self.ndim = self.solutions[0].size if len(self.solutions) > 0 else None

    # def _generate_new_population(self, ):
    #     self.solutions = []
    #     for p in range(self.popsize):
    #         # Generalise solution creation to any type of genome
    #         indiv = create_learning_rule(self._genome_type, **self._genome_params)
    #         self.solutions.append(indiv)

    # def reset(self):
    #     self._generate_new_population()

    def ask(self) -> List:
        if self._first_gen:
            self._generate_new_population()
        else:
            self._generate_offspring()
        return super().ask()

    def tell(self, fitnesses, *, gen_no: int = None):
        # assert len(fitnesses) == len(self.solutions)
        # self.fitnesses = fitnesses
        # if not self.minimise:
        #     fitnesses = -fitnesses
        # idx_best = np.argsort(fitnesses)[:self.mu]
        # self.best_fitness = fitnesses[idx_best[0]]
        # self.best_solution = self.take_solutions(idx_best[0])
        super().tell(fitnesses, gen_no=gen_no)

        # TODO: Deal with fitness ties between parent and offspring (choose offspring)
        if not self.minimise:
            fitnesses = -fitnesses
        idx_best = np.argsort(fitnesses)[:self.mu]
        self.parents = self.take_solutions(idx_best)

        # Moved offspring generation to self.ask()
        # self._generate_offspring()

    def _generate_offspring(self):
        assert (self.parents is not None) or (len(self.parents) > 0)
        self.solutions = [*self.parents]

        for i in range(self.lambd):
            # Sample from parent (if more than one, select randomly)
            if self.mu > 1:
                idx = np.random.randint(0, self.mu)
            else:
                idx = 0
            p = self.parents[idx]
            # If using genome object's mutation implementation
            if self.use_genome_operator:
                offspring = p.mutate(rate=self.mutation_rate, scale=self.mutation_scale, method=self.mutation_method)
            # Otherwise, use algorithm's own implementation
            else:
                p_genome = p.parameters
                o_genome = self._mutate_array(p_genome)
                # Rebuild phenotype after mutation
                offspring = self._create_individual(genome=o_genome)

            self.solutions.append(offspring)

    # def _mutate_array(self, ind: np.ndarray) -> np.ndarray:
    #     """Perform mutation when genome is a 1D array"""
    #     if self.mutation_method == "resample":
    #         flags = np.random.binomial(1, p=self.mutation_rate, size=ind.size).astype(bool)
    #         new_p = np.random.uniform(low=self._sampling_low, high=self._sampling_high, size=ind.size)
    #         o = np.where(flags, new_p, ind)
    #     elif self.mutation_method == "perturb":
    #         flags = np.random.binomial(1, p=self.mutation_rate, size=ind.size).astype(bool)
    #         noise = np.random.normal(loc=0, scale=self.mutation_scale, size=ind.size)
    #         o = np.where(flags, ind + noise, ind)
    #         o = np.clip(o, self.lower_bounds, self.upper_bounds)
    #     else:
    #         raise NotImplementedError(f"Mutation method {self.mutation_method} not supported.")
    #     return o

    def _mutate_array(self, ind: np.ndarray) -> np.ndarray:
        """Perform mutation when genome is a 1D array (Mutate each position step-by-step)"""
        off_genome = ind.copy()

        for i, dtype in zip(range(off_genome.size), self._gene_dtypes):
            # Decide whether to mutate this gene or not
            if np.random.rand() > self.mutation_rate:
                continue
            # Mutation
            if self.mutation_method == "resample":
                if dtype is int:
                    new_p = np.random.randint(self._sampling_low[i], self._sampling_high[i], size=1)
                elif dtype is float:
                    new_p = np.random.uniform(self._sampling_low[i], self._sampling_high[i], size=1)
                else:
                    raise TypeError(f"dtype={dtype} not recognised for gene position {i}")

            elif self.mutation_method == "perturb":
                old_p = off_genome[i]
                if dtype is int:
                    # TODO: Incorporate mutation_scale into integer noise
                    noise = np.random.randint(-1, 1 + 1, size=1)
                elif dtype is float:
                    noise = np.random.normal(loc=0, scale=self.mutation_scale, size=1)
                else:
                    raise TypeError(f"dtype={dtype} not recognised for gene position {i}")
                new_p = old_p + noise
                new_p = np.clip(new_p, self.lower_bounds[i], self.upper_bounds[i])
            else:
                raise NotImplementedError(f"Mutation method {self.mutation_method} not supported.")
            off_genome[i] = new_p
        return off_genome

    def __repr__(self):
        return f"MuPlusLambda(ndim={self.ndim}, mu={self.mu}, lambd={self.lambd}, minimise={self.minimise}, genome_type={self._genome_type})"