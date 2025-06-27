import numpy as np
from typing import Literal, Protocol, Tuple, override, Union
from collections import deque


class RewarderProtocol(Protocol):
    # def calculate_reward(self, target) -> float:
    #     pass

    # def calculate_fitness(self, label: int, reward: float) -> float:
    #     pass

    # def get_fitness(self) -> float:
    #     pass
    def reset(self):
        """Reset the rewarder state."""
        pass
    def get_target(self, current_class: int) -> np.ndarray:
        """Get the target spikes for the current class."""
        pass
    def get_reward(self, target_spikes: np.ndarray, output_spikes: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the reward based on the target spikes and output spikes.
        
        Returns:
            Tuple[float, float]: (error, reward)
        """

class CollectorProtocol(Protocol):
    def record(self, reward: float, error: float) -> None:
        """Record the reward and error at each time step."""
        pass

    def collate(self):
        """
        Internally combine the collected rewards and errors within each buffer interval (sample).
        """
        pass

    def calculate_fitness(self) -> float:
        """
        Calculate the fitness based on the collected rewards and errors.
        
        Returns:
            float: The calculated fitness value.
        """
        pass


def fitness_func(error, a=3, b=2):
    return a / (1 + error) - b

class SimpleRewarder(RewarderProtocol):
    """
    Compare against pre-generated target array instead of comparing each output spikes timestep-by-timestep.
    """
    def __init__(self, num_classes, pattern_length, *,
                 spacing: int = None,
                 target_position: Literal["first", "last"] = "last"):
        self.num_classes = num_classes
        self.pattern_length = pattern_length
        self.full_length = pattern_length
        self.spacing = spacing if spacing is not None else 0
        self.target_position = target_position
        self._label = 0 if target_position == "first" else pattern_length - 1
        self._create_target_array()
        self.reset()

    def _create_target_array(self):
        self.target_array = np.zeros((self.num_classes, self.num_classes, self.pattern_length), dtype=np.int_)
        for i in range(self.num_classes):
            self.target_array[i, i, self._label] = 1
        if self.spacing > 0:
            self.target_array = np.pad(self.target_array, ((0, 0), (0, 0), (0, self.spacing)), mode='constant', constant_values=0)
            self.full_length = self.target_array.shape[2] - 1

    def reset(self):
        self.count = 0

    def get_target(self, current_class):
        idx = self.count % self.full_length

        target_spikes = self.target_array[current_class, :, idx]
        if self.count >= self.full_length:
            self.count = 0
        
        self.count += 1
        return target_spikes
    
    def get_reward(self, target_spikes, output_spikes):
        error = np.sum(np.abs(target_spikes - output_spikes))
        reward = fitness_func(error)
        return error, reward
    

class SimpleCollector(CollectorProtocol):
    def __init__(self, buffer_size: int = None, fitness_type: Literal["reward", "error"] = "reward"):
        self.rewards = []
        self.errors = []
        self.reward_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        self.fitness_type = fitness_type

    def reset(self):
        self.rewards.clear()
        self.errors.clear()
        self.reward_buffer.clear()
        self.error_buffer.clear()

    def record(self, reward: float, error: float) -> None:
        self.reward_buffer.append(reward)
        self.error_buffer.append(error)

    def collate(self):
        self.rewards.append(sum(self.reward_buffer))
        self.errors.append(sum(self.error_buffer))
        self.reward_buffer.clear()
        self.error_buffer.clear()

    def calculate_fitness(self) -> float:
        if self.fitness_type == "reward":
            return sum(self.rewards)
        elif self.fitness_type == "error":
            return sum(self.errors)
        else:
            return 0.0