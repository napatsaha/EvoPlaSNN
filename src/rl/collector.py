import numpy as np
from collections import namedtuple
from typing import List


EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "length", "exploration"])


class RewardCollector:
    records: List[EpsInfo]
    def __init__(self, *, cutoff_timestep: int = 0):
        # self.reward_history = []
        # self.episode_lengths = []
        self.records = []
        self.cutoff_timestep = cutoff_timestep
        self.minimise = False
        
    def reset(self):
        """
        Reset the collector's history.
        """
        # self.reward_history.clear()
        # self.episode_lengths.clear()
        self.records.clear()

    def collect(self, t: int, episode: int, reward: float, episode_length: int, exploration: float):
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
            exploration = exploration
        ))

    def get_rewards(self, use_cutoff: bool = False) -> list[float]:
        if use_cutoff and self.cutoff_timestep > 0:
            return [r.reward for r in self.records if r.t >= self.cutoff_timestep]
        else:
            return [r.reward for r in self.records]
    
    def get_episode_lengths(self, use_cutoff: bool = False) -> list[int]:
        if use_cutoff and self.cutoff_timestep > 0:
            return [r.length for r in self.records if r.t >= self.cutoff_timestep]
        else:
            return [r.length for r in self.records]
    
    def get_explorations(self, use_cutoff: bool = False) -> list[float]:
        if use_cutoff and self.cutoff_timestep > 0:
            return [r.exploration for r in self.records if r.t >= self.cutoff_timestep]
        else:
            return [r.exploration for r in self.records]
    
    def get_fitness(self):
        """
        Sum of all rewards collected.  
        (Assuming fixed simulation length, larger sum means shorter episode.)
        """
        return np.mean(self.get_rewards(use_cutoff=True))