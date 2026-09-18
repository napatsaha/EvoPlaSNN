import numpy as np
from collections import namedtuple
from typing import List, Literal, Dict, Callable


EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "eps_reward", "length", "starting_state", "exploration", "truncated", "terminated", "trajectory"])


class RewardCollector:
    records: List[EpsInfo]
    # _valid_fitness_types: List[str] = ["reward", "latency"]
    _optimise_direction_dict: dict = {
        "reward": 'maximise',
        "eps_reward": "maximise",
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
        "eps_reward": (-np.inf, np.inf),
        "success_rate": (0.0, 1.0),
        "latency": (0.0, np.inf)
    }
    def __init__(self, *, fitness_type: Literal["reward", "latency", "success_rate", "eps_reward"] = "reward",
                 fitness_agg_func: Literal["mean", "sum", "min", "max", "median"] = "mean",
                #  max_fitness: float = 1.0, min_fitness: float = -1.0,
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
        self.max_fitness = self._bounds.get(self.fitness_type)[1]
        self.min_fitness = self._bounds.get(self.fitness_type)[0]
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

    def collect(self, t: int = None, episode: int = None, reward: float = None, 
                eps_reward: float = None, episode_length: int = None, 
                starting_state: int = None, exploration: float = None, 
                truncated: bool = False, terminated: bool = False, trajectory: 'Trajectory' = None):
        """
        Record final reward and total step count at the end of an episode.
        """
        # self.reward_history.append(reward)
        # self.episode_lengths.append(episode_length)
        self.records.append(EpsInfo(
            t = t,
            episode = episode,
            reward = reward,
            eps_reward = eps_reward,
            length = episode_length,
            starting_state = starting_state,
            exploration = exploration,
            truncated = truncated,
            terminated = terminated,
            trajectory = trajectory
        ))

    def get_rewards(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if t_cutoff is not None:
            return [r.reward for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.reward for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.reward for r in self.records]

    def get_eps_rewards(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if t_cutoff is not None:
            return [r.eps_reward for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.eps_reward for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.eps_reward for r in self.records]
    
    def get_episode_lengths(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[int]:
        if t_cutoff is not None:
            return [r.length for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.length for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.length for r in self.records]
    
    def get_timestamps(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[int]:
        if t_cutoff is not None:
            return [r.t for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.t for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.t for r in self.records]

    def get_explorations(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if t_cutoff is not None:
            return [r.exploration for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.exploration for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.exploration for r in self.records]
    
    def get_success(self, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if t_cutoff is not None:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.episode >= eps_cutoff]
        else:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records]

    def get_intermediate_fitness(self, t_cutoff: int = None, eps_cutoff: int = None) -> List[float]:
        """
        Return list of episode fitnesses before aggregation.
        """
        if self.fitness_type == "reward":
            fitnesses = self.get_rewards(t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "eps_reward":
            fitnesses = self.get_eps_rewards(t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "success_rate":
            fitnesses = self.get_success(t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "latency":
            fitnesses = self.get_episode_lengths(t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        else:
            fitnesses = []
        return fitnesses

    def get_fitness(self, t_cutoff: int = None, eps_cutoff: int = None) -> float:
        """
        Calculate fitness for current trial. Depends on `fitness_type`
        """
        fitnesses = self.get_intermediate_fitness(t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        if len(fitnesses) == 0:
            return self.min_fitness
        agg = self._agg_func_dict.get(self.fitness_agg_func)
        return agg(fitnesses)
    

Trajectory = namedtuple("Trajectory", ["t", "state", "observation", "action", "reward", "done", "info"])

class TrajectoryCollector:
    records: List[Trajectory]
    def __init__(self, *, default_record: bool = True,
                 record_t: bool = None, record_obs: bool = None, record_action: bool = None, record_reward: bool = None,
                 record_done: bool = None, record_state: bool = None, record_info = None,
                 ):
        self.records = []
        self.default_record = bool(default_record)
        self._record_t = record_t if record_t is not None else self.default_record
        self._record_obs = record_obs if record_obs is not None else self.default_record
        self._record_action = record_action if record_action is not None else self.default_record
        self._record_reward = record_reward if record_reward is not None else self.default_record
        self._record_done = record_done if record_done is not None else self.default_record
        self._record_state = record_state if record_state is not None else self.default_record
        self._record_info = record_info if record_info is not None else self.default_record

    def reset(self):
        self.records.clear()

    def collect(self, locs: dict):
        self.records.append(Trajectory(
            t = locs.get("t", None) if self._record_t else None,
            state = locs.get("info").get("current_state", None) if self._record_state else None,
            observation= locs.get("state") if self._record_obs else None,
            action= locs.get("action") if self._record_action else None,
            reward= locs.get("reward") if self._record_reward else None,
            done= locs.get("episode_done") if self._record_done else None,
            info= locs.get("info") if self._record_info else None
        ))

    # def collect(self, t = None, observation = None, action = None, reward = None, done = None, info = None, state = None):
    #     self.records.append(Trajectory(
    #         t=t,
    #         state = state,
    #         observation = observation,
    #         action = action,
    #         reward = reward,
    #         done = done,
    #         info = info
    #     ))

    def get_summed_rewards(self) -> float:
        if not self._record_reward:
            return None
        return sum([rec.reward for rec in self.records if rec.reward is not None])