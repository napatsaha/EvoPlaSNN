import numpy as np
from collections import namedtuple
from typing import List


EpsInfo = namedtuple("EpsInfo", ["t", "episode", "reward", "length", "exploration"])


class RewardCollector:
    records: List[EpsInfo]
    def __init__(self):
        # self.reward_history = []
        # self.episode_lengths = []
        self.records = []
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

    def get_rewards(self) -> list[float]:
        return [r.reward for r in self.records]
    
    def get_episode_lengths(self) -> list[int]:
        return [r.length for r in self.records]
    
    def get_explorations(self) -> list[float]:
        return [r.exploration for r in self.records]
    
    def get_fitness(self):
        """
        Sum of all rewards collected.  
        (Assuming fixed simulation length, larger sum means shorter episode.)
        """
        return sum(self.get_rewards())