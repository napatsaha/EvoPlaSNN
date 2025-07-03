from functools import partial
import logging
from pathlib import Path
import jax
import jax.numpy as jnp

from evosax.types import Params, Population
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems.problem import Problem, State

from evo.base import Evaluator


class ProblemWrapper(Problem):
    """
    Wrapper for Problem to use with EvoSax.
    Converts default Evaluator interface to EvoSax's Problem interface.
    """
    evaluator: Evaluator
    def __init__(self, evaluator: Evaluator, num_trials: int = None):
        self.evaluator = evaluator
        self.num_trials = num_trials
        self._num_dims = evaluator.get_parameter_size()
        self._minimise = evaluator.is_minimise()

    def init(self, key: jax.Array) -> State:
        return State(counter=0)

    def sample(self, key: jax.Array):
        return jax.random.uniform(
            key, 
            shape=(self._num_dims,), 
            minval=-1.0, 
            maxval=1.0
        )

    # @partial(jax.jit, static_argnames=("self", ))
    def eval(self, key: jax.Array, population: Population, state: State):
        fitnesses = []
        for solution in population:
            fitness = self.evaluator.evaluate(solution, num_trials=self.num_trials)
            # Force minimisation because evasax algorithms only deals with minimisation problems.
            if self._minimise:
                fitness = -fitness
            fitnesses.append(fitness)
        fitnesses = jnp.array(fitnesses)
        state = state.replace(counter=state.counter + 1)
        info = {}
        return fitnesses, state, info

    @property
    def num_dims(self):
        return self._num_dims


class EvoManager:
    """
    Main class for managing loop of evolutionary optimisation using EvoSax.
    """
    def __init__(self, solver: EvolutionaryAlgorithm, evaluator: Problem, *, 
                 num_trials: int = 1, log_file: str = None, logging_freq: int = 1,
                 max_generations: int = 1000, target_fitness: float = 0.0, tolerance: float = 1e-6, save_best: int = 1,
                 num_stagnations: int = 100, stag_tolerance: float = 1e-8
                 ):

        self.num_trials = num_trials
        self.save_best = save_best
        self.log_file = log_file
        self.logging_freq = logging_freq

        # Optimsation parameters
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.tolerance = tolerance
        self.num_stagnations = num_stagnations
        self.stag_tolerance = stag_tolerance

        self.solver = solver
        self.evaluator = evaluator

        # solver = CMA_ES(population_size=pop_size, solution=np.array(dummy_solution))

        self.params = solver.default_params

    def _setup_logger(self):
        """
        Set up channels for outputting logging information.
        By default, prints to console.
        """
        self.logger = logging.getLogger("EvoManager")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler() if self.log_file is None else logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if self.logger.hasHandlers():
            # Prevent adding multiple handlers if logger is already configured
            self.logger.handlers.clear()
        self.logger.addHandler(handler)

    def run(self, seed: int = 0):
        self._setup_logger()
        if hasattr(self.evaluator, "num_trials") and self.evaluator.num_trials is None:
            self.evaluator.num_trials = self.num_trials

        # Initialisation
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        dummy_solution = self.evaluator.sample(subkey)
        state = self.solver.init(subkey, dummy_solution, self.params)
        problem_state = self.evaluator.init(subkey)

        stag_count = 0
        previous_best = None
        metrics_log = []
        for i in range(self.max_generations):
            key, subkey = jax.random.split(key)
            key_ask, key_eval, key_tell = jax.random.split(subkey, 3)

            population, state = self.solver.ask(key_ask, state, self.params)
            fitness, problem_state, info = self.evaluator.eval(key_eval, population, problem_state)
            state, metrics = self.solver.tell(key_tell, population, fitness, state, self.params)

            metrics_log.append(metrics)

            best_fitness = metrics["best_fitness"]
            best_solution = metrics["best_solution"]

            # Log result
            if i % self.logging_freq == 0:
                self.logger.info(f"Generation {i}: Best fitness = {best_fitness:.3f}, Best solution = {best_solution.round(2)}")

            # Check stopping criteria
            if jnp.abs(best_fitness - self.target_fitness) < self.tolerance:
                # Reached target fitness
                self.logger.info(f"Target fitness {self.target_fitness} reached at generation {i}.")
                break

            if i >= self.max_generations:
                # Reached maximum generations
                self.logger.info(f"Maximum generations {self.max_generations} reached.")
                break
            
            if previous_best is not None:
                if jnp.abs(best_fitness - previous_best) < self.stag_tolerance:
                    stag_count += 1
                else:
                    stag_count = 0

            if previous_best is None or best_fitness < previous_best:
                previous_best = best_fitness

            if stag_count >= self.num_stagnations:
                print(f"Stopping early at generation {i} due to stagnation.")
                break

        self.logger.info("Terminating Evolutionary optimisation.")
        self.logger.info(f"Best solution: {best_solution.round(4)}")
        self.logger.info(f"Best fitness: {best_fitness:.3f}")

        # Save best solution
        if self.log_file is not None:
            save_path = Path(self.log_file).parent
            self.solver.save_best(save_path, n=self.save_best, precision=6)

        return metrics_log