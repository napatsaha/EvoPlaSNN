# from lrule.cgp import CGP_Graph, CGP_Rule
from lrule.utils import create_learning_rule
from .base import BaseSolver, Genome
from typing import List

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

    def __init__(self, mu, lambd, *, ndim = 2, popsize = None, minimise = True,
                 lrule_type: str = None, lrule_params: dict = None,
                 **kwargs):
        # super().__init__(ndim, popsize, minimise)
        self.minimise = minimise
        self.mu = mu
        self.lambd = lambd
        self.popsize = self.mu + self.lambd

        self._lrule_params = kwargs if lrule_params is None else lrule_params
        if lrule_type is not None:
            self._lrule_type = lrule_type
        else:
            if "type" in self._lrule_params:
                self._lrule_type = self._lrule_params.pop("type")
        self._generate_new_population()
        self.ndim = self.solutions[0].size

    def _generate_new_population(self, ):
        self.solutions = []
        for p in range(self.popsize):
            # Generalise solution creation to any type of genome
            indiv = create_learning_rule(self._lrule_type, **self._lrule_params)
            self.solutions.append(indiv)
        self.parents = []

    # def reset(self):
    #     self._generate_new_population()

    def ask(self):
        return self.solutions

    def tell(self, fitnesses):
        assert len(fitnesses) == len(self.solutions)
        self.fitnesses = fitnesses
        if not self.minimise:
            fitnesses = -fitnesses
        idx_best = np.argsort(fitnesses)[:self.mu]
        self.best_fitness = fitnesses[idx_best[0]]
        self.best_solution = self.take_solutions(idx_best[0])

        # TODO: Deal with fitness ties between parent and offspring (choose offspring)
        self.parents = self.take_solutions(idx_best)

        self._generate_offspring()

    def _generate_offspring(self):
        assert (self.parents is not None) or (len(self.parents) > 0)
        self.solutions = [*self.parents]

        for i in range(self.lambd):
            # Sample from parent
            if self.mu > 1:
                idx = np.random.randint(0, self.mu)
            else:
                idx = 0
            offspring = self.parents[idx].mutate()
            self.solutions.append(offspring)