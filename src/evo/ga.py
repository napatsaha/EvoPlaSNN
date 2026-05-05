from common.base import Solver, Genome
from .base import BaseSolver
from typing import Literal


class GeneticAlgorithm(BaseSolver):
    def __init__(self, ndim = 2, popsize = None, minimise = True, *, 
                 genome_type = None, genome_params = None,
                 selection: Literal["tournament", "rank", "roulette"],
                 mutation_rate: float = 0.1):
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)

    def ask(self):
        return super().ask()
    
    def tell(self, fitnesses):
        return super().tell(fitnesses)
    
