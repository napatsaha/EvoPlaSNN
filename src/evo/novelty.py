from collections import namedtuple
from typing import List

import numpy as np
from scipy.spatial import distance as dst

from .base import BaseSolver
from common.base import LearningRule
from common.utils import make_input_grid, assymetric_min_max_normalise


def compute_bc(rule: LearningRule, inp: np.ndarray = None, normalise: bool = False, *,
                           num_bin=10, bounds=None) -> np.ndarray:
    assert isinstance(rule, LearningRule), "Currently only support Behaviour Characterisation for LearningRule"
    assert inp.ndim == 2
    assert inp.shape[1] == rule.input_size, "Second dimension of input arrays should have the same size as rule's inputs"

    if inp is None:
        if bounds is None:
            raise NotImplementedError(f"Auto-calculating Input Bounds from Rule alone is not yet supported")
        inp = make_input_grid(bounds, num_bin)

    bc = rule.forward(inp)
    if normalise:
        bc = assymetric_min_max_normalise(bc)
    return bc.flatten()


ArchiveEntry = namedtuple('ArchiveEntry', ["rule", "bc", "fitness"])


class NoveltySearchLC(BaseSolver):
    """
    Novelty Search with Local Competition (NSLC) + NSGA-II
    """
    def __init__(self, bounds, num_grid, k: int, novelty_threshold: float,
                 ndim = 2, popsize = None, minimise = True, *, 
                 genome_type = None, genome_params = None):
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)
        self.bounds = bounds
        self.num_grid = num_grid
        self.k = k
        self.novelty_threshold = novelty_threshold
        self._inp = make_input_grid(bounds, num_grid)
        self.bcs = []
        self.novelty_dist = None

        self.archive: List[ArchiveEntry] = []

    def ask(self):
        if self._first_gen:
            self._generate_new_population()
        # Compute Behaviour Characterisation (doesn't require fitnesses)
        self.bcs = []
        for sol in self.solutions:
            bc = compute_bc(sol, self._inp, normalise=False)
            self.bcs.append(bc)
        # Return solutions
        return super().ask()
    
    def tell(self, fitnesses):

        # Calculate novelty distance
        # novelty_dist = dst.squareform(dst.pdist(np.asarray(self.bcs), metric="cosine"))
        # Average novelty metric and Local Competition
        rhos = []
        lcs = []
        # Concatenate current population with archive
        bcs = [sol.bc for sol in self.archive] + self.bcs
        fts = [sol.fitness for sol in self.archive] + list(fitnesses)

        for i in range(self.popsize):
            # Find nearest individuals based on novelty distance
            # nov_dist_i = novelty_dist[:, i]
            bc_i = self.bcs[i]
            nov_dist_i = [dst.cosine(bc_i, bc) for bc in bcs]
            k_nearest = np.argsort(nov_dist_i)[1:self.k+1]
            # Novelty metric = Average novelty distance to k nearest neighbours
            rho = np.mean(np.take(nov_dist_i, k_nearest))
            rhos.append(rho)
            # Local Quality metric = How many k nearest neighbour perform wose than this individual
            ft_i = fitnesses[i]
            nearest_fts = [fts[ix] for ix in k_nearest]
            highest_fts_than_i = [ft < ft_i if self.minimise else ft > ft_i for ft in nearest_fts]
            lc = np.mean(highest_fts_than_i)
            lcs.append(lc)

            # Add to archive
            if rho > self.novelty_threshold:
                sol_info = ArchiveEntry(
                    rule=self.solutions[i],
                    bc=self.bcs[i],
                    fitness=ft_i
                )
                self.archive.append(sol_info)

        

        return rhos, lcs
    
    