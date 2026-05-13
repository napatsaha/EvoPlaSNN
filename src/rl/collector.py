import numpy as np
from collections import namedtuple
from typing import List, Literal, Dict, Callable


EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "length", "starting_state", "exploration", "truncated", "terminated", "trajectory"])


class RewardCollector:
    records: List[EpsInfo]
    # _valid_fitness_types: List[str] = ["reward", "latency"]
    _optimise_direction_dict: dict = {
        "reward": 'maximise',
        "success_rate": 'maximise',
        "latency": 'minimise'
    }
    _agg_func_dict: Dict[str, Callable] = {
        "mean": np.mean,
        "max": np.maximum,
        "min": np.minimum,
        "median": np.median,
        "sum": np.sum
    }
    _bounds = {
        "reward": (-1.0, 1.0),
        "success_rate": (0.0, 1.0),
        "latency": (0.0, np.inf)
    }
    def __init__(self, *, fitness_type: Literal["reward", "latency"] = "reward",
                 fitness_agg_func: Literal["mean", "sum", "min", "max", "median"] = "mean",
                 max_fitness: float = 1.0, min_fitness: float = -1.0,
                 fitness_on_eval_only: bool = True):
        """
        Args:
            fitness_type (str, optional): Method of calculating fitness in an episode. Currently supports: ["reward", "latency"].
                Default: reward
            fitness_agg_func (str, optional): Function to aggregate episode fitnesses. Currently supports:
                ["mean", "sum", "min", "max", "median"]. Default: mean.
            fitness_on_eval_only (bool, optional): Whether to include episodes during phase for fitness calculations. 
                Note that this is controlled externally (i.e. a simulator needs to call `soft_reset()` of this class after training).
                Default: True
        """
        # self.reward_history = []
        # self.episode_lengths = []
        self.records = []
        if fitness_type not in self._optimise_direction_dict:
            raise ValueError(f"Fitness type {fitness_type} is not supported. Only accepts {self._optimise_direction_dict.keys()}")
        self.fitness_type = fitness_type
        self.minimise = self._optimise_direction_dict.get(self.fitness_type) == 'minimise'
        if fitness_agg_func not in self._agg_func_dict:
            if not isinstance(fitness_agg_func, Callable):
                raise ValueError(f"Aggregate function {fitness_agg_func} not supported. \
                                 Only accepts {self._agg_func_dict.keys()} or a Callable.")
        self.fitness_agg_func = fitness_agg_func
        # TODO: Get min-max fitness from environment
        self.max_fitness = max_fitness
        self.min_fitness = min_fitness
        self.fitness_on_eval_only = fitness_on_eval_only

    def reset(self):
        """
        Reset the collector's history.
        """
        # self.reward_history.clear()
        # self.episode_lengths.clear()
        self.records.clear()

    def soft_reset(self):
        """
        Reset after a training phase. Behaviour depends on `fitness_on_eval_only`
        """
        if self.fitness_on_eval_only:
            self.reset()

    def collect(self, t: int, episode: int, reward: float, episode_length: int, 
                starting_state: int = None, exploration: float = None, 
                truncated: bool = False, terminated: bool = False, trajectory = None):
        """
        Record final reward and total step count at the end of an episode.
        """
        # self.reward_history.append(reward)
        # self.episode_lengths.append(episode_length)
        self.records.append(EpsInfo(
            t = t,
            episode = episode,
            reward = reward,
            length = episode_length,
            starting_state = starting_state,
            exploration = exploration,
            truncated = truncated,
            terminated = terminated,
            trajectory = trajectory
        ))

    def get_rewards(self, cutoff: int = None) -> list[float]:
        if cutoff is not None:
            return [r.reward for r in self.records if r.t >= cutoff]
        else:
            return [r.reward for r in self.records]
    
    def get_episode_lengths(self, cutoff: int = None) -> list[int]:
        if cutoff is not None:
            return [r.length for r in self.records if r.t >= cutoff]
        else:
            return [r.length for r in self.records]
    
    def get_timestamps(self, cutoff: int = None) -> list[int]:
        if cutoff is not None:
            return [r.t for r in self.records if r.t >= cutoff]
        else:
            return [r.t for r in self.records]

    def get_explorations(self, cutoff: int = None) -> list[float]:
        if cutoff is not None:
            return [r.exploration for r in self.records if r.t >= cutoff]
        else:
            return [r.exploration for r in self.records]
    
    def get_success(self, cutoff: int = None) -> list[float]:
        if cutoff is not None:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.t >= cutoff]
        else:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records]

    def get_intermediate_fitness(self, cutoff: int = None) -> List[float]:
        """
        Return list of episode fitnesses before aggregation.
        """
        if self.fitness_type == "reward":
            fitnesses = self.get_rewards(cutoff=cutoff)
        elif self.fitness_type == "success_rate":
            fitnesses = self.get_success(cutoff=cutoff)
        elif self.fitness_type == "latency":
            fitnesses = self.get_episode_lengths(cutoff=cutoff)
        else:
            fitnesses = []
        return fitnesses

    def get_fitness(self, cutoff: int = None) -> float:
        """
        Calculate fitness for current trial. Depends on `fitness_type`
        """
        fitnesses = self.get_intermediate_fitness(cutoff)
        if len(fitnesses) == 0:
            return self.min_fitness
        agg = self._agg_func_dict.get(self.fitness_agg_func)
        return agg(fitnesses)
    

Trajectory = namedtuple("Trajectory", ["state", "observation", "action", "reward", "done", "info"])

class TrajectoryCollector:
    records: List[Trajectory]
    def __init__(self):
        self.records = []

    def reset(self):
        self.records.clear()

    def collect(self, observation, action, reward, done, info, state = None):
        self.records.append(Trajectory(
            state = state,
            observation = observation,
            action = action,
            reward = reward,
            done = done,
            info = info
        ))
