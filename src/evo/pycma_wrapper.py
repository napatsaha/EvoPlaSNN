"""
BaseSolver Wrapper for pycma.CMAEvolutionStrategy
"""

import numpy as np
from cma import CMAEvolutionStrategy
import cma

from .base import BaseSolver
from common.utils import create_learning_rule


class PyCMAWrapper(BaseSolver):
    def __init__(self, ndim = 2, popsize = None, minimise = True, *, 
                 x0 = None, sigma0 = 1.0,
                 mutation_method = "resample", mutation_rate = 0.5, mutation_scale = 1, 
                 genome_type = None, genome_params = None,
                 **kwargs):
        super().__init__(ndim, popsize, minimise, mutation_method=mutation_method, mutation_rate=mutation_rate, mutation_scale=mutation_scale, 
                         genome_type=genome_type, genome_params=genome_params)
        sample_sol = self._create_individual()
        self.ndim = sample_sol.size

        if x0 is None:
            x0 = np.zeros(self.ndim)
        else:
            assert len(x0) == self.ndim

        self.model = CMAEvolutionStrategy(
            x0 = x0,
            sigma0=sigma0,
            options=dict(
                popsize=self.popsize,
                **kwargs
            )
        )

    def reset(self):
        return super().reset()

    def setup_logger(self, log_path = None):
        super().setup_logger(log_path)
        if self.log_path is not None:
            save_dir = self.log_path / "outcmaes"
            save_dir.mkdir()
            self.model.logger.name_prefix = save_dir.absolute().as_posix() + "/"

    def ask(self):
        sols = self.model.ask()
        self.solutions = sols
        return super().ask()

    def tell(self, fitnesses, *, gen_no = None):
        super().tell(fitnesses, gen_no=gen_no)

        if not self.minimise:
            fitnesses = -self.fitnesses

        self.model.tell(self.solutions, fitnesses)
        self.model.logger.add(modulo=1)

    def wrapup(self, n_best, precision = None):
        self.model.stop()
        self.model.plot()
        cma.s.figsave(self.log_path / "pycma_plot.png")
        # cma.s.figclose()
        return super().wrapup(n_best, precision)

    def close(self):
        return super().close()
