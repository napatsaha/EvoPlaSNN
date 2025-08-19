import numpy as np


class RewardCollector:
    def __init__(self):
        self.reward_history = []
        self.episode_lengths = []
        self.minimise = False
        
    def reset(self):
        """
        Reset the collector's history.
        """
        self.reward_history.clear()
        self.episode_lengths.clear()

    def collect(self, reward: float, episode_length: int):
        """
        Record final reward and total step count at the end of an episode.
        """
        self.reward_history.append(reward)
        self.episode_lengths.append(episode_length)

    def get_rewards(self) -> list[float]:
        return self.reward_history
    
    def get_episode_lengths(self) -> list[int]:
        return self.episode_lengths
    
    def get_fitness(self):
        """
        Sum of all rewards collected.  
        (Assuming fixed simulation length, larger sum means shorter episode.)
        """
        return sum(self.reward_history)