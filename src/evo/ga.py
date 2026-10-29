from common.base import Solver, Genome
from .base import BaseSolver
from typing import Literal, List, Dict

import numpy as np


class GeneticAlgorithm(BaseSolver):
    def __init__(self, ndim = None, popsize = None, minimise = True, *, 
                 genome_type = None, genome_params = None,
                 selection: Literal["tournament", "rank", "roulette"] = "tournament",
                 tournament_size: int = 2,
                 crossover_rate: float = 0.5,
                 mutation_method: Literal["resample", "perturb"] = "resample",
                 mutation_rate: float = 0.1, mutation_scale: float = 1.0):
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)
        self.selection = selection
        self.tournament_size = int(tournament_size)
        self.crossover_rate = crossover_rate
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def ask(self):
        if self._first_gen:
            # For initialising random population
            self._generate_new_population()
        else:
            self._generate_next_generation()
        return super().ask() # Call BaseSolver.ask() last, so that `_first_gen` is set to False after initialisation
    
    def tell(self, fitnesses, *, gen_no: int = None):
        super().tell(fitnesses, gen_no=gen_no)

    def _generate_next_generation(self):
        # Select parents for next generation
        pop = self._select_parents()
        # Crossover parents
        np.random.shuffle(pop)
        pop = self._crossover(pop)

        # Mutate offsprings of crossover genome
        pop = self._mutate(pop)

        # Set new population to solutions
        self.solutions = pop

    def _select_parents(self) -> List[Genome]:
        if self.selection == "tournament":
            return self._tournament_select(self.solutions, self.fitnesses, self.tournament_size)
        elif self.selection == "roulette":
            return self._roulette_select(self.solutions, self.fitnesses)
        elif self.selection == "random":
            return self._random_select(self.solutions)

    def _tournament_select(self, sols: List[Genome], fits: np.ndarray, size: int) -> List[Genome]:
        pop = []
        n = len(sols)
        for i in range(n):
            subpop_idx = np.random.choice(np.arange(n), size=size, replace=False)
            subpop_fts = fits[subpop_idx]
            best_idx = np.argmin(subpop_fts) if self.minimise else np.argmax(subpop_fts)
            winner_idx: int = int(subpop_idx[best_idx])
            winner = sols[winner_idx]
            pop.append(winner)
        return pop

    def _roulette_select(self, sols, fits) -> List[Genome]:
        raise NotImplementedError()

    def _random_select(self, sols) -> List[Genome]:
        raise NotImplementedError()

    def _crossover(self, inter_pop: List[Genome]) -> List[Genome]:
        pop = []
        n = len(inter_pop)
        for i in range(0, n, 2):
            par1, par2 = inter_pop[i], inter_pop[i+1]
            child1 = par1.crossover(par2, rate=self.crossover_rate)
            child2 = par2.crossover(par1, rate=self.crossover_rate)
            pop.append(child1)
            pop.append(child2)
        return pop

    def _mutate(self, pop: List[Genome]) -> List[Genome]:
        pop_m = []
        for ind in pop:
            ind_m = ind.mutate(method=self.mutation_method, rate=self.mutation_rate, scale=self.mutation_scale)
            pop_m.append(ind_m)
        return pop_m

    
    def __repr__(self):
        return f"GeneticAlgorithm(ndim={self.ndim}, popsize={self.popsize}, minimise={self.minimise}, genome_type={self._genome_type})"

