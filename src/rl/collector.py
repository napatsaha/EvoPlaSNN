import numpy as np
from collections import namedtuple
from typing import List, Literal, Dict, Callable, NamedTuple, Any, Tuple


# Trajectory = namedtuple("Trajectory", ["t", "state", "observation", "action", "reward", "done", "info"])

class Trajectory(NamedTuple):
    t: int = None
    state: int = None
    observation: np.ndarray = None
    action: int = None
    reward: float = None
    done: bool = None
    info: Dict[str, Any] = None

    __slot__ = ()

    def __repr__(self):
        return "EpsInfo(" + \
            ", ".join("%s=%s" % (key, val) for key, val in self._asdict().items() if val is not None) + \
            ")"

# EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "eps_reward", "length", "starting_state", "exploration", "truncated", "terminated", "trajectory"])

class EpsInfo(NamedTuple):
    t: int = None
    episode: int = None
    season: int = None
    reward: float = None
    eps_reward: float = None
    length: int = None
    starting_state: int = None
    exploration: np.ndarray | float = None
    truncated: bool = None
    terminated: bool = None
    trajectory: Trajectory = None
    collect: bool = None

    __slot__ = ()

    def __repr__(self):
        return "EpsInfo(" + \
            ", ".join("%s=%s" % (key, val) for key, val in self._asdict().items() if val is not None) + \
            ")"

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
                 fitness_t_cutoff: int = None, fitness_eps_cutoff: int = None,
                 fitness_season_t_cutoff: int = None, fitness_season_eps_cutoff: int = None,
                #  max_fitness: float = 1.0, min_fitness: float = -1.0,
                 fitness_on_eval_only: bool = True):
        """
        Args:
            fitness_type (str, optional): Method of calculating fitness in an episode. Currently supports: ["reward", "latency"].
                Default: reward
            fitness_agg_func (str, optional): Function to aggregate episode fitnesses. Currently supports:
                ["mean", "sum", "min", "max", "median"]. Default: mean.
            fitness_t_cutoff (int, optional): Only calculate fitness after this simulation timestep. Defaults to None.
            fitness_eps_cutoff (int, optional): Only calculate fitness after this simulation episode. Defaults to None.
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

        # Own cutoff when calculating fitness
        self.fitness_t_cutoff = fitness_t_cutoff
        self.fitness_eps_cutoff = fitness_eps_cutoff
        self.fitness_season_t_cutoff = fitness_season_t_cutoff
        self.fitness_season_eps_cutoff = fitness_season_eps_cutoff

        # Season counter (for calculating seasonal fitness cutoff)
        self._last_season = None
        self._last_season_t = None
        self._last_season_eps = None

    def _check_collect_fitness(self, t, eps):
        """
        Determine (based on cutoff method, t and eps) whether this episode should be included in final fitness calculation

        Args:
            t (_type_): timestep
            eps (_type_): episode number
        """
        if self.fitness_t_cutoff is not None:
            return t >= self.fitness_t_cutoff
        if self.fitness_eps_cutoff is not None:
            return eps >= self.fitness_eps_cutoff
        if self.fitness_season_t_cutoff is not None:
            return (t-self._last_season_t) >= self.fitness_season_t_cutoff
        if self.fitness_season_eps_cutoff is not None:
            return (eps-self._last_season_eps) >= self.fitness_season_eps_cutoff

    def reset(self):
        """
        Reset the collector's history.
        """
        # self.reward_history.clear()
        # self.episode_lengths.clear()
        self.records.clear()
        self._last_season = None
        self._last_season_t = None
        self._last_season_eps = None

    def soft_reset(self):
        """
        Reset after a training phase. Behaviour depends on `fitness_on_eval_only`
        """
        if self.fitness_on_eval_only:
            self.reset()

    def collect(self, t: int = None, episode: int = None, season: int = None,
                reward: float = None, eps_reward: float = None, episode_length: int = None, 
                starting_state: int = None, exploration: float = None, 
                truncated: bool = False, terminated: bool = False, trajectory: Trajectory = None):
        """
        Record episode information at the end of an episode.
        """
        if (self._last_season is None) or (self._last_season != season):
            self._last_season = season
            self._last_season_t = t
            self._last_season_eps = episode
        
        collect = self._check_collect_fitness(t, episode)
        self.records.append(EpsInfo(
            t = t,
            episode = episode,
            season=season,
            reward = reward,
            eps_reward = eps_reward,
            length = episode_length,
            starting_state = starting_state,
            exploration = exploration,
            truncated = truncated,
            terminated = terminated,
            trajectory = trajectory,
            collect=collect
        ))

    def get_rewards(self, collect_only: bool = True, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if collect_only:
            return [r.reward for r in self.records if r.collect]
        elif t_cutoff is not None:
            return [r.reward for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.reward for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.reward for r in self.records]

    def get_eps_rewards(self, collect_only: bool = True, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if collect_only:
            return [r.eps_reward for r in self.records if r.collect]
        elif t_cutoff is not None:
            return [r.eps_reward for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.eps_reward for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.eps_reward for r in self.records]
    
    def get_episode_lengths(self, collect_only: bool = True, t_cutoff: int = None, eps_cutoff: int = None) -> list[int]:
        if collect_only:
            return [r.length for r in self.records if r.collect]
        elif t_cutoff is not None:
            return [r.length for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [r.length for r in self.records if r.episode >= eps_cutoff]
        else:
            return [r.length for r in self.records]

    def get_season_timestamps(self) -> List[Tuple[int, int]]:
        return [(r.t, r.season) for i, r in enumerate(self.records[:]) if (i==0) or (i > 0 and self.records[i-1].season != r.season)]
    
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
    
    def get_success(self, collect_only: bool = True, t_cutoff: int = None, eps_cutoff: int = None) -> list[float]:
        if collect_only:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.collect]
        elif t_cutoff is not None:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.t >= t_cutoff]
        elif eps_cutoff is not None:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records if r.episode >= eps_cutoff]
        else:
            return [1 if r.terminated and r.reward == self.max_fitness else 0 for r in self.records]

    def get_intermediate_fitness(self, collect_only: bool = True, t_cutoff: int = None, eps_cutoff: int = None) -> List[float]:
        """
        Return list of episode fitnesses before aggregation.
        """
        if self.fitness_type == "reward":
            fitnesses = self.get_rewards(collect_only=collect_only, t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "eps_reward":
            fitnesses = self.get_eps_rewards(collect_only=collect_only, t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "success_rate":
            fitnesses = self.get_success(collect_only=collect_only, t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        elif self.fitness_type == "latency":
            fitnesses = self.get_episode_lengths(collect_only=collect_only, t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        else:
            fitnesses = []
        return fitnesses

    def get_fitness(self, t_cutoff: int = None, eps_cutoff: int = None) -> float:
        """
        Calculate fitness for current trial, based on `fitness_type`.  
        If `t_cutoff` or `eps_cutoff` is not specified, `fitness_t_cutoff` or `fitness_eps_cutoff` will be used, respectively (if specified during init).  
        t_cutoff always have priority over eps_cutoff, if both are given.
        """
        # if t_cutoff is None and self.fitness_t_cutoff is not None:
        #     t_cutoff = self.fitness_t_cutoff
        # if eps_cutoff is None and self.fitness_eps_cutoff is not None:
        #     eps_cutoff = self.fitness_eps_cutoff
        fitnesses = self.get_intermediate_fitness(collect_only=True, t_cutoff=t_cutoff, eps_cutoff=eps_cutoff)
        if len(fitnesses) == 0:
            return self.min_fitness
        agg = self._agg_func_dict.get(self.fitness_agg_func)
        return agg(fitnesses)
    


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