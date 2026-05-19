from collections import namedtuple
from typing import List, Tuple

import numpy as np
from scipy.spatial import distance as dst

from .base import BaseSolver
from . import nsga2
from common.base import Genome, LearningRule
from common.utils import make_input_grid, assymetric_min_max_normalise


def compute_lrule_bc(rule: LearningRule, inp: np.ndarray = None, normalise: bool = False, *,
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


class NoveltySearch(BaseSolver):
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
            bc = compute_lrule_bc(sol, self._inp, normalise=False)
            self.bcs.append(bc)
        # Return solutions
        return super().ask()
    
    def tell(self, fitnesses):

        # Calculate novelty distance
        # novelty_dist = dst.squareform(dst.pdist(np.asarray(self.bcs), metric="cosine"))
        # Average novelty metric and Local Competition
        rhos = [] # Average novelty distance
        lcs = [] # Local competition
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
    
    
# Solution = namedtuple("Solution", ("genome", "novelty_dist", "local_fitness", "global_fitness", "behaviour"), 
#                       defaults=(None, 0.0, 0.0, 0.0, None))

class Solution:
    def __init__(self, genome=None, novelty_dist=None, local_fitness=None, global_fitness=None, behaviour=None):
        self.genome = genome
        self.novelty_dist = novelty_dist
        self.local_fitness = local_fitness
        self.global_fitness = global_fitness
        self.behaviour = behaviour

    def __repr__(self):
        return f"Solution(genome={repr(self.genome)}, novelty_dist={self.novelty_dist:.2g}, local_fitness={self.local_fitness:.2g}, " + \
            f"global_fitness={np.round(self.global_fitness, 3)}, behaviour={np.round(self.behaviour, 2)})"

class NSLC(BaseSolver):

    def __init__(self, bounds, num_grid, k: int, novelty_threshold: float = None, archive_prob: float = 0.05,
                 ndim = None, popsize = None, *, 
                 crossover_rate: float = 0.5, mutation_rate: float = 0.5,
                 normalise: bool = False,
                 genome_type = None, genome_params = None):
        minimise = False
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)
        self.crossover_rate = np.clip(crossover_rate, 0, 1)
        self.mutation_rate = np.clip(mutation_rate, 0, 1)
        self.normalise = normalise

        self.bounds = bounds
        self.num_grid = num_grid
        self.k = k
        self.novelty_threshold = novelty_threshold
        self.archive_prob = archive_prob
        self.archive_method = "threshold" if self.novelty_threshold is not None else "probabilistic"
        self._inp = make_input_grid(bounds, num_grid)
        # self.bcs = []
        # self.novelty_dist = None

        self.archive: List[Solution] = []
        self.parents: List[Solution] = []
        # self.parent_novs = []
        # self.parent_lfts = []


    def _generate_offspring(self):
        if self.parents is None:
            raise ValueError("No parents to create offsprings from")
        Q = []
        for i in range(self.popsize):
            # Tournament selection
            ix, iy = np.random.choice(self.popsize, size=2, replace=False)
            Px = self.parents[ix].genome
            Py = self.parents[iy].genome
            offspring = Px.crossover(Py, rate=self.crossover_rate)
            offspring = offspring.mutate(rate=self.mutation_rate)
            Q.append(offspring)
        return Q
    
    def _compute_bc(self, solutions):
        bcs = []
        for sol in solutions:
            bc = compute_lrule_bc(sol, self._inp, normalise=self.normalise)
            bcs.append(bc)
        return bcs

    def _eval_novelty(self, fitnesses, bcs) -> Tuple[List, List]:
        assert len(fitnesses) == len(bcs)
        N = len(fitnesses)
        rhos = [] # Average novelty distance
        lcs = [] # Local competition
        # Concatenate current population with archive
        bcs = [sol.behaviour for sol in self.archive] + bcs
        fts = [sol.global_fitness for sol in self.archive] + list(fitnesses)

        for i in range(N):
            # Find nearest individuals based on novelty distance
            # nov_dist_i = novelty_dist[:, i]
            bc_i = bcs[i]
            nov_dist_i = [dst.cosine(bc_i, bc) for bc in bcs]
            k_nearest = np.argsort(nov_dist_i)[1:self.k+1]
            # Novelty metric = Average novelty distance to k nearest neighbours
            rho = np.mean(np.take(nov_dist_i, k_nearest))
            rhos.append(-rho)
            # Local Quality metric = How many k nearest neighbour perform wose than this individual
            ft_i = fitnesses[i]
            nearest_fts = [fts[ix] for ix in k_nearest]
            highest_fts_than_i = [ft < ft_i if self.minimise else ft > ft_i for ft in nearest_fts]
            lc = np.mean(highest_fts_than_i)
            lcs.append(-lc)

            # # Add to archive
            # if rho > self.novelty_threshold:
            #     sol_info = ArchiveEntry(
            #         rule=self.solutions[i],
            #         bc=bcs[i],
            #         fitness=ft_i
            #     )
            #     self.archive.append(sol_info)

        return rhos, lcs

    def _make_new_generation(self, R, fronts_R, ranks_R, o1_R, o2_R):
        idx_parents = []

        n = 0
        for f in fronts_R:
            n_front = len(f)
            # n += n_front
            if n + n_front <= self.popsize:
                idx_parents.extend(f)
                n += n_front
            else:
                dist = nsga2.crowding_distance_single_front(f, len(R), o1_R, o2_R)
                dist_f = np.take(dist, f)
                sorted_f = np.take(f, np.argsort(-dist_f))
                idx_parents.extend(sorted_f[:(self.popsize - n)])
                break
    
        # P = np.take(R, idx_parents, axis=0)
        # o1 = np.take(o1_R, idx_parents)
        # o2 = np.take(o2_R, idx_parents)

        return idx_parents

    def _add_to_archive(self):
        if self.archive_method == "threshold":
            for sol in self.parents:
                if -sol.novelty_dist >= self.novelty_threshold:
                    self.archive.append(sol)
        elif self.archive_method == "probabilistic":
            for sol in self.parents:
                if np.random.rand() < self.archive_prob:
                    self.archive.append(sol)

    def ask(self):
        if self._first_gen:
            self._generate_new_population()
            sols = self.solutions
            self.parents = [Solution(genome=sol) for sol in self.solutions]
        else:
            sols = self._generate_offspring()
            self.solutions = sols
        return sols
    
    def tell(self, fitnesses):
        self.fitnesses = fitnesses

        bcs = self._compute_bc(self.solutions)
        novs, lfts = self._eval_novelty(fitnesses, bcs)

        if self._first_gen:
            # If first generation, no offspring is created yet
            # self.parent_novs = novs
            # self.parent_lfts = lfts
            for ix, p in enumerate(self.parents):
                p.novelty_dist = novs[ix]
                p.local_fitness = lfts[ix]
                p.behaviour = bcs[ix]
                p.global_fitness = fitnesses[ix]
            self._first_gen = False

        else:
            # Combine parent and offspring population
            R = [sol.genome for sol in self.parents] + self.solutions
            # novs_R = np.concatenate([self.parent_novs, novs])
            # lfts_R = np.concatenate([self.parent_lfts, lfts])
            fts_R = [sol.global_fitness for sol in self.parents] + list(fitnesses)
            bcs_R = [sol.behaviour for sol in self.parents] + list(bcs)
            novs_R = [sol.novelty_dist for sol in self.parents] + list(novs)
            lfts_R = [sol.local_fitness for sol in self.parents] + list(lfts)

            fronts_R, ranks_R = nsga2.fast_nondominated_sort(R, novs_R, lfts_R)
            assert len(R) == sum(len(f) for f in fronts_R)
            idx_parents = self._make_new_generation(
                R, fronts_R, ranks_R, novs_R, lfts_R
            )

            self.parents.clear()
            self.parents = [
                Solution(genome=R[ix],
                         novelty_dist=novs_R[ix],
                         local_fitness=lfts_R[ix],
                         global_fitness=fts_R[ix],
                         behaviour=bcs_R[ix])
                         for ix in idx_parents
            ]

        self._add_to_archive()