# from lrule.cgp import CGP_Graph, CGP_Rule
from common.base import Genome, LearningRule
from common.utils import create_learning_rule
from genome.genome import SimpleGenome
from .base import BaseSolver
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

    def __init__(self, mu, lambd, *, minimise = True,
                #  mutation_rate: float = 0.1,
                #  lrule_type: str = None, lrule_params: dict = None,
                 **kwargs):
        """
        Mu + Lambda Algorithm

        Args:
            mu (int): Number of parents
            lambd (int): Number of offsprings
            minimise (bool, optional): Optimisation direction. Defaults to True.
            mutation_rate (float, optional): Probability of mutating each gene. Defaults to 0.1.
        """
        # self.minimise = minimise
        self.mu = mu
        self.lambd = lambd
        popsize = self.mu + self.lambd
        # self.mutation_rate = np.clip(0, 1, mutation_rate)

        super().__init__(ndim=None, popsize=popsize, minimise=minimise,
                          **kwargs)


        # self._lrule_params = kwargs if lrule_params is None else lrule_params
        # if lrule_type is not None:
        #     self._lrule_type = lrule_type
        # else:
        #     if "type" in self._lrule_params:
        #         self._lrule_type = self._lrule_params.pop("type")
        self.parents: List[Genome] = []

        self.ndim = self.solutions[0].size if len(self.solutions) > 0 else None

    def _generate_new_population(self, ):
        self.solutions = []
        for p in range(self.popsize):
            # Generalise solution creation to any type of genome
            indiv = create_learning_rule(self._genome_type, **self._genome_params)
            self.solutions.append(indiv)

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
        super().tell(fitnesses)

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
            # Sample from parent
            if self.mu > 1:
                idx = np.random.randint(0, self.mu)
            else:
                idx = 0
            offspring = self.parents[idx].mutate(rate=self.mutation_rate, scale=self.mutation_scale, method=self.mutation_method)
            if not isinstance(offspring, LearningRule):
                if self._genome_type is not None:
                    offspring = create_learning_rule(self._genome_type, parameters=offspring, **self._genome_params)
                else:
                    offspring = SimpleGenome(parameters=offspring)
            self.solutions.append(offspring)

    def __repr__(self):
        return f"MuPlusLambda(ndim={self.ndim}, mu={self.mu}, lambd={self.lambd}, minimise={self.minimise})"