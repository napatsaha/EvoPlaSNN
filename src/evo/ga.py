from common.base import Solver, Genome
from .base import BaseSolver
from typing import Literal, List, Dict, Tuple
from common.utils import create_learning_rule

import numpy as np


class GeneticAlgorithm(BaseSolver):
    def __init__(self, ndim = None, popsize = None, minimise = True, *, 
                 genome_type = None, genome_params = None,
                 use_genome_operator: bool = True, default_lower_bound: float = 0.0, default_upper_bound: float = 1.0,
                 selection: Literal["tournament", "rank", "roulette"] = "tournament",
                 tournament_size: int = 2,
                 n_elites: int = 0,
                 crossover_method: Literal["uniform"] = "uniform",
                 crossover_rate: float = 0.5,
                 mutation_method: Literal["resample", "perturb"] = "resample",
                 mutation_rate: float = 0.1, mutation_scale: float = 1.0):
        """
        Simple Genetic Algorithm.

        Warning: 
        1. If using `use_genome_operator=True`, crossover is not true crossover, in that the two offsprings produced are not the result of
        swapping genome, but rather two independent calls to the parents' `crossover()` method.
        2. If using `use_genome_operator=False`, i.e. using this class's genetic operator, we do not yet take into account the dtype of object
        (e.g. discrete param) when performing mutation, nor do we sample anything other than uniformly between given bounds. If a gene is unbounded, 
        either in the lower or upper bounds, or both, the bounds will be replaced with `default_lower_bound` and `default_upper_bound` respectively.
        There is no guarantee the default bound will be compatible with the other bound (if given)
        3. `crossover_rate` should remain at 0.5. Any other rates simply mean the crossover process favours one parent over another, which would produce
        uneven offsprings.
        4. `mutation_scale` only affects perturb-type mutation (the standard deviation of the zero-centered Gaussian noise). Meanwhile, `mutation_rate` 
        affects both types of mutation.

        Args:
            ndim (_type_, optional): _description_. Defaults to None.
            popsize (_type_, optional): _description_. Defaults to None.
            minimise (bool, optional): _description_. Defaults to True.
            genome_type (_type_, optional): _description_. Defaults to None.
            genome_params (_type_, optional): _description_. Defaults to None.
            use_genome_operator (bool, optional): _description_. Defaults to True.
            default_lower_bound (float, optional): _description_. Defaults to 0.0.
            default_upper_bound (float, optional): _description_. Defaults to 1.0.
            selection (Literal[&quot;tournament&quot;, &quot;rank&quot;, &quot;roulette&quot;], optional): _description_. Defaults to "tournament".
            tournament_size (int, optional): _description_. Defaults to 2.
            n_elites (int, optional): _description_. Defaults to 0.
            crossover_method (Literal[&quot;uniform&quot;], optional): _description_. Defaults to "uniform".
            crossover_rate (float, optional): _description_. Defaults to 0.5.
            mutation_method (Literal[&quot;resample&quot;, &quot;perturb&quot;], optional): _description_. Defaults to "resample".
            mutation_rate (float, optional): _description_. Defaults to 0.1.
            mutation_scale (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__(ndim, popsize, minimise, genome_type=genome_type, genome_params=genome_params)
        self.use_genome_operator = use_genome_operator
        self.selection = selection
        self.tournament_size = int(tournament_size)
        self.crossover_method = crossover_method
        self.crossover_rate = crossover_rate
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        assert 0 <= n_elites < self.popsize, f"`n_elites` must be a positive integer less than popsize={self.popsize}. " + f"Got `n_elites={n_elites}`"
        self.n_elites = int(n_elites)
        if not self.use_genome_operator:
            self.lower_bounds, self.upper_bounds = self.sample_sol.get_bounds()
            # For use only in sampling in mutation_method="resample"
            self._sampling_low = np.where(np.array(self.lower_bounds) == None, default_lower_bound, self.lower_bounds).astype(float)
            self._sampling_high = np.where(np.array(self.upper_bounds) == None, default_upper_bound, self.upper_bounds).astype(float)
            # Replace None with -inf/+inf for use in clipping
            self.lower_bounds = np.where(np.array(self.lower_bounds) == None, -np.inf, self.lower_bounds).astype(float)
            self.upper_bounds = np.where(np.array(self.upper_bounds) == None, np.inf, self.upper_bounds).astype(float)


    def ask(self):
        if self._first_gen:
            # For initialising random population
            self._generate_new_population()
        else:
            self.solutions = self._generate_next_generation()
        return super().ask() # Call BaseSolver.ask() last, so that `_first_gen` is set to False after initialisation
    
    def tell(self, fitnesses, *, gen_no: int = None):
        super().tell(fitnesses, gen_no=gen_no)

    def _generate_next_generation(self) -> List[Genome]:
        pop = []
        if self.n_elites > 0:
            elites = self._select_best_n(n=self.n_elites)
            pop.extend(elites)

        # n_off = self.popsize - self.n_elites
        n_left = int(self.popsize - self.n_elites) # Counter to determine when there is odd places left
        while len(pop) < self.popsize:
            p1, p2 = self._select_parents()

            if self.use_genome_operator:
                # Rely on each Genome object's crossover() and mutate() methods
                # Crossover to produce two offsprings
                o1, o2 = self._crossover_genome(p1, p2)

                if n_left > 1:
                    # Mutate each offspring
                    o1 = self._mutate_genome(o1)
                    o2 = self._mutate_genome(o2)
                    pop.extend([o1, o2])
                else:
                    o1 = self._mutate_genome(o1)
                    pop.append(o1)
            else:
                # Otherwise, extract own genome array and perform operations
                # Extract genome.parameters as arrays
                p1a = p1.parameters
                p2a = p2.parameters
                # Crossover to produce two offsprings
                o1, o2 = self._crossover_array(p1a, p2a)
                if n_left > 1:
                    # Mutate each offspring
                    o1 = self._mutate_array(o1)
                    o2 = self._mutate_array(o2)
                    # Reconstruct Genome from parameters array
                    o1 = create_learning_rule(self._genome_type, parameters=o1, **self._genome_params)
                    o2 = create_learning_rule(self._genome_type, parameters=o2, **self._genome_params)
                    pop.extend([o1, o2])
                else:
                    o1 = self._mutate_array(o1)
                    o1 = create_learning_rule(self._genome_type, parameters=o1, **self._genome_params)
                    pop.append(o1)

            n_left -= 2

        return pop

        # # Select parents for next generation
        # pop = self._select_parents()
        # # Crossover parents
        # np.random.shuffle(pop)
        # pop = self._crossover_pop(pop)

        # # Mutate offsprings of crossover genome
        # pop = self._mutate_pop(pop)

        # Set new population to solutions
        # self.solutions = pop

    def _select_parents(self) -> List[Genome]:
        p1i = self._select_one_index()
        p2i = p1i # Start with identical index
        while p1i == p2i: # Loop until second parent is different
            p2i = self._select_one_index()
        return self.solutions[p1i], self.solutions[p2i]

    def _select_one_index(self) -> int:
        if self.selection == "tournament":
            return self._tournament_select()
        elif self.selection == "roulette":
            return self._roulette_select()
        elif self.selection == "random":
            return self._random_select()

    def _tournament_select(self) -> int:
        """
        Return index of one best individual from a random sample of size `tournament_size`

        Returns:
            int: Individual index
        """
        subpop_idx = np.random.choice(np.arange(self.popsize), size=self.tournament_size, replace=False)
        subpop_fts = self.fitnesses[subpop_idx]
        best_idx = np.argmin(subpop_fts) if self.minimise else np.argmax(subpop_fts)
        winner_idx: int = int(subpop_idx[best_idx])
        return winner_idx

    def _roulette_select(self) -> int:
        raise NotImplementedError()

    def _random_select(self) -> int:
        raise NotImplementedError()

    def _crossover_genome(self, p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
        child1 = p1.crossover(p2, rate=self.crossover_rate)
        child2 = p2.crossover(p1, rate=self.crossover_rate)
        return child1, child2

    def _mutate_genome(self, ind: Genome) -> Genome:
        return ind.mutate(method=self.mutation_method, rate=self.mutation_rate, scale=self.mutation_scale)

    def _crossover_array(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # raise NotImplementedError("Custom array-based crossover will be implemented in the future")
        assert p1.size == p2.size, "Both parents must have the same genome size"
        if self.crossover_method == "uniform":
            flags = np.random.binomial(1, p=self.crossover_rate, size=p1.size).astype(bool)
            o1 = np.where(flags, p1, p2)
            o2 = np.where(~flags, p1, p2)
        else:
            raise NotImplementedError(f"Crossover method {self.crossover_method} not supported.")
        return o1, o2

    def _mutate_array(self, ind: np.ndarray) -> np.ndarray:
        # raise NotImplementedError("Custom array-based mutation will be implemented in the future")
        if self.mutation_method == "resample":
            flags = np.random.binomial(1, p=self.mutation_rate, size=ind.size).astype(bool)
            new_p = np.random.uniform(low=self._sampling_low, high=self._sampling_high, size=ind.size)
            o = np.where(flags, new_p, ind)
        elif self.mutation_method == "perturb":
            flags = np.random.binomial(1, p=self.mutation_rate, size=ind.size).astype(bool)
            noise = np.random.normal(loc=0, scale=self.mutation_scale, size=ind.size)
            o = np.where(flags, ind + noise, ind)
            o = np.clip(o, self.lower_bounds, self.upper_bounds)
        else:
            raise NotImplementedError(f"Mutation method {self.mutation_method} not supported.")
        return o

    # def _crossover_pop(self, inter_pop: List[Genome]) -> List[Genome]:
    #     pop = []
    #     n = len(inter_pop)
    #     for i in range(0, n, 2):
    #         par1, par2 = inter_pop[i], inter_pop[i+1]
    #         child1 = par1.crossover(par2, rate=self.crossover_rate)
    #         child2 = par2.crossover(par1, rate=self.crossover_rate)
    #         pop.append(child1)
    #         pop.append(child2)
    #     return pop

    # def _mutate_pop(self, pop: List[Genome]) -> List[Genome]:
    #     pop_m = []
    #     for ind in pop:
    #         ind_m = ind.mutate(method=self.mutation_method, rate=self.mutation_rate, scale=self.mutation_scale)
    #         pop_m.append(ind_m)
    #     return pop_m

    
    def __repr__(self):
        return f"GeneticAlgorithm(ndim={self.ndim}, popsize={self.popsize}, minimise={self.minimise}, genome_type={self._genome_type})"

