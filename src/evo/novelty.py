from collections import namedtuple
from typing import List, Literal, Sequence, Tuple, Callable
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import distance as dst

from .base import BaseSolver
from . import nsga2
from common.base import Genome
from common.utils import make_input_grid, compute_lrule_bc


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
    """
    Container object for recording genome along with auxiliary information in Novelty Search with Local Competition:
    - genome
    - novelty_dist
    - local_fitness
    - global_fitness
    - behaviour
    - rank
    """
    def __init__(self, genome=None, novelty_dist=None, local_fitness=None, global_fitness=None, behaviour=None, rank=None):
        self.genome: Genome = genome
        self.novelty_dist: float = novelty_dist
        self.local_fitness: float = local_fitness
        self.global_fitness: float = global_fitness
        self.behaviour: ArrayLike = behaviour
        self.rank: int = rank

    def __repr__(self):
        return f"Solution(genome={np.round(self.genome.parameters, 2)}, novelty_dist={self.novelty_dist:.2g}, local_fitness={self.local_fitness:.2g}, " + \
            f"global_fitness={np.round(self.global_fitness, 3)}, behaviour={np.round(self.behaviour, 2)}, rank={self.rank})"

class NSLC(BaseSolver):

    def __init__(self, popsize: int, k: int, minimise: bool = True, #bc_func: Callable, 
                 novelty_threshold: float = None, archive_prob: float = 0.05,
                 dist_metric: Literal["cosine", "euclidean"] = "cosine",
                 ndim = None, *, 
                 crossover_rate: float = 0.5, mutation_rate: float = 0.5,
                 genome_type = None, genome_params = None, **kwargs):
        # minimise = False
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)
        self.crossover_rate = np.clip(crossover_rate, 0, 1)
        self.mutation_rate = np.clip(mutation_rate, 0, 1)
        # self.normalise = normalise
        # self.bounds = bounds
        # self.num_grid = num_grid
        self.k = k
        self.novelty_threshold = novelty_threshold
        self.archive_prob = archive_prob
        self.archive_method = "threshold" if self.novelty_threshold is not None else "probabilistic"
        # self._inp = make_input_grid(bounds, num_grid)
        # self.bcs = []
        # self.novelty_dist = None

        # assert isinstance(bc_func, Callable), f"bc_func must be a `Callable` taking in a solution as input. Got type={type(bc_func)}"
        # self.bc_func: Callable = bc_func

        self.dist_metric = dist_metric
        self.dist_func = getattr(dst, self.dist_metric)

        self.archive: List[Solution] = []
        self.parents: List[Solution] = []
        self.offsprings: List[Solution] = []
        # self.parent_novs = []
        # self.parent_lfts = []

    def reset(self) -> None:
        super().reset()
        self.archive.clear()
        self.parents.clear()
        self.offsprings.clear()

    def ask(self):
        if self._first_gen:
            self._generate_new_population()
            sols = self.solutions
            self.parents = [Solution(genome=sol) for sol in self.solutions]
        else:
            sols = self._generate_offspring()
            self.solutions = sols
            self.offsprings = [Solution(genome=sol) for sol in self.solutions]
        return sols
    
    def tell(self, fitnesses: List[float], behaviours: List[ArrayLike]) -> None:
        super().tell(fitnesses)

        # bcs = self._compute_bc(self.solutions)

        # If first generation, record evaluation results as that of parents
        if self._first_gen:
            # Record evaluation results: fitness and behaviour
            for ix, sol in enumerate(self.parents):
                sol.behaviour = behaviours[ix]
                sol.global_fitness = fitnesses[ix]
            # Evaluate Novelty and Local Quality within Parents
            novs, lfts = self._eval_novelty(self.parents)

            fronts, ranks = nsga2.fast_nondominated_sort(-novs, -lfts)

            for ix, sol in enumerate(self.parents):
                sol.novelty_dist = novs[ix]
                sol.local_fitness = lfts[ix]
                sol.rank = ranks[ix]
            
            self._first_gen = False

        # Otherwise, the evaluation results corresponds to the offspring solutions
        else:
            # Record evaluation results: fitness and behaviour
            for ix, sol in enumerate(self.offsprings):
                sol.behaviour = behaviours[ix]
                sol.global_fitness = fitnesses[ix]

            # Combine parent and offspring population
            R = self.parents + self.offsprings
            # Evaluate Novelty and Local Quality within (Parents + Offsprings)
            novs, lfts = self._eval_novelty(R)
            fronts_R, ranks_R = nsga2.fast_nondominated_sort(-novs, -lfts)

            for ix, sol in enumerate(R):
                sol.novelty_dist = novs[ix]
                sol.local_fitness = lfts[ix]
                sol.rank = ranks_R[ix]

            # Select new parents based on Lower Rank -> Higher Novelty
            idx_parents = self._sort_and_select(
                N=self.popsize, fronts=fronts_R, dist=novs
            )

            # self.parents.clear()
            self.parents = [
                sol for ix, sol in enumerate(R) if ix in idx_parents
            ]

        self._add_to_archive()


    def _generate_offspring(self):
        if self.parents is None:
            raise ValueError("No parents to create offsprings from")
        Q = []
        for _ in range(self.popsize):
            # Tournament selection
            # TODO: Make it a true Tournament Selection:
            # -> Save fronts and ranks from fast-sort parents
            ix, iy = np.random.choice(self.popsize, size=2, replace=False)
            Px = self.parents[ix].genome
            Py = self.parents[iy].genome
            offspring = Px.crossover(Py, rate=self.crossover_rate)
            offspring = offspring.mutate(rate=self.mutation_rate)
            Q.append(offspring)
        return Q
    
    # def _compute_bc(self, solutions):
    #     bcs = []
    #     for sol in solutions:
    #         bc = self.bc_func(sol)
    #         bcs.append(bc)
    #     return bcs

    def _eval_novelty(self, solutions: List[Solution]) -> Tuple[List, List]:
        """
        Calculate average novelty distance and local quality to k nearest neighbour, of the list of 
        'solutions' combined with archive.

        Novelty will be calculated from average distance among behaviour space to k nearest neighbout,
        according to `NSLC.dist_metric`

        Local Quality will be the proportion among those k nearest neighbour with worse fitness than each solution
        (depending on whether `NSLC.minimise` is True)

        Args:
            solutions (List[Solution]): List of solutions to evaluate. Each member must have 'global_fitness' 
            and 'behaviour' attributes recorded

        Returns:
            Tuple[List, List]: Novelty, Local fitness
        """
        # assert len(fitnesses) == len(bcs)
        # N = len(fitnesses)
        rhos = [] # Average novelty distance (Diversity)
        lcs = [] # Local competition (Quality)
        # Concatenate current population with archive
        bcs = [sol.behaviour for sol in solutions] + [sol.behaviour for sol in self.archive]
        fts = [sol.global_fitness for sol in solutions] + [sol.global_fitness for sol in self.archive]

        for i, sol in enumerate(solutions):
            # Find nearest individuals based on novelty distance
            bc_i = sol.behaviour
            nov_dist_i = [self.dist_func(bc_i, bc) for j, bc in enumerate(bcs) if i != j]
            k_nearest = np.argsort(nov_dist_i)[1:self.k+1]
            # Novelty metric = Average novelty distance to k nearest neighbours
            rho = np.mean(np.take(nov_dist_i, k_nearest))
            rhos.append(rho) # Maximising novelty

            # Local Quality metric = How many k nearest neighbour perform wose than this individual
            ft_i = sol.global_fitness
            nearest_fts = [fts[ix] for ix in k_nearest]
            better_than_neighbour = [ft > ft_i if self.minimise else ft < ft_i for ft in nearest_fts]
            lc = np.mean(better_than_neighbour)
            lcs.append(lc)

            # # Add to archive
            # if rho > self.novelty_threshold:
            #     sol_info = ArchiveEntry(
            #         rule=self.solutions[i],
            #         bc=bcs[i],
            #         fitness=ft_i
            #     )
            #     self.archive.append(sol_info)

        return np.asarray(rhos), np.asarray(lcs)

    def _sort_and_select(self, N, fronts, dist):
        idx_parents = []

        n = 0
        for f in fronts:
            n_front = len(f)
            # n += n_front
            if n + n_front <= N:
                idx_parents.extend(f)
                n += n_front
            else:
                # dist = nsga2.crowding_distance_single_front(f, len(R), o1_R, o2_R)
                # Instead of crowding distance, use a custom distance argument
                dist_f = np.take(dist, f)
                sorted_f = np.take(f, np.argsort(-dist_f)) # Take top-n indivs within front with largest dist
                idx_parents.extend(sorted_f[:(N - n)])
                break
    
        # P = np.take(R, idx_parents, axis=0)
        # o1 = np.take(o1_R, idx_parents)
        # o2 = np.take(o2_R, idx_parents)

        return idx_parents

    def _add_to_archive(self):
        if self.archive_method == "threshold":
            for sol in self.parents:
                if sol.novelty_dist >= self.novelty_threshold:
                    self.archive.append(deepcopy(sol))
        elif self.archive_method == "probabilistic":
            for sol in self.parents:
                if np.random.rand() < self.archive_prob:
                    self.archive.append(deepcopy(sol))


    def __repr__(self):
        return f"NSLC(k={self.k}, ndim={self.ndim}, popsize={self.popsize}, minimise={self.minimise})"
