import numpy as np
from collections import namedtuple
from typing import List


EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "length", "exploration", "truncated", "terminated", "trajectory"])


class RewardCollector:
    records: List[EpsInfo]
    def __init__(self, *, max_fitness: float = 1.0, min_fitness: float = -1.0):
        # self.reward_history = []
        # self.episode_lengths = []
        self.records = []
        self.minimise = False
        self.max_fitness = max_fitness
        self.min_fitness = min_fitness
        
    def reset(self):
        """
        Reset the collector's history.
        """
        # self.reward_history.clear()
        # self.episode_lengths.clear()
        self.records.clear()

    def collect(self, t: int, episode: int, reward: float, episode_length: int, exploration: float = None,
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
    
    def get_fitness(self, cutoff: int = None) -> float:
        """
        Sum of all rewards collected.  
        (Assuming fixed simulation length, larger sum means shorter episode.)
        """
        rewards = self.get_rewards(cutoff=cutoff)
        if len(rewards) == 0:
            return self.min_fitness
        return np.mean(rewards)
    

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
