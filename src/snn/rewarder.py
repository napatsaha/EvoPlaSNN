import numpy as np
from typing import Literal, Protocol, Tuple, Callable
from collections import deque

def create_rewarder(type: str, **kwargs):
    if type == "simple":
        class_name = "SimpleRewarder"
    # return getattr(globals(), class_name)(**kwargs)
    return globals()[class_name](**kwargs)

def create_collector(type: str, **kwargs):
    if type == "simple":
        class_name = "SimpleCollector"
    # return getattr(globals(), class_name)(**kwargs)
    return globals()[class_name](**kwargs)


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

    @property
    def minimise(self) -> bool:
        """
        Whether the fitness should be minimised.
        
        Returns:
            bool: True if minimising, False if maximising.
        """
        return self._minimise


def fitness_func(error, a=3, b=2):
    return a / (1 + error) - b

class SimpleRewarder(RewarderProtocol):
    """
    Compare against pre-generated target array instead of comparing each output spikes timestep-by-timestep.
    """
    def __init__(self, num_classes, pattern_length, *,
                 spacing: int = None,
                 target_position: Literal["first", "last"] = "last", **kwargs):
        self.num_classes = num_classes
        self.pattern_length = pattern_length
        self.full_length = pattern_length
        self.spacing = spacing if spacing is not None else 0
        self.target_position = target_position
        self._label = 0 if target_position == "first" else pattern_length - 1
        self._create_target_array()
        self.reset()

    def _create_target_array(self):
        # Create pre-generated array of target outputs for easier slicing
        self.target_array = np.zeros((self.num_classes, self.num_classes, self.pattern_length), dtype=np.int_)
        for i in range(self.num_classes):
            self.target_array[i, i, self._label] = 1
        if self.spacing > 0:
            # Add spacing after pattern
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
        # Sum the absolute differences between target and output spikes
        error = np.sum(np.abs(target_spikes - output_spikes))
        # Reward = +1 for Error = 0,
        # Reward = -0.5 for Error = 1,
        # Reward = -1 for Error = 2, etc
        reward = fitness_func(error)
        return error, reward
    

class SimpleCollector(CollectorProtocol):
    def __init__(self, buffer_size: int = None, fitness_type: Literal["reward", "error"] = "reward", agg_func: Callable | str = "sum",
                 **kwargs):
        self.rewards = []
        self.errors = []
        self.reward_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        if fitness_type not in ["reward", "error"]:
            raise ValueError("fitness_type must be either 'reward' or 'error'")
        self.fitness_type = fitness_type
        self._minimise = True if fitness_type == "error" else False
        if isinstance(agg_func, str):
            if agg_func == "sum":
                self.agg_func = sum
            elif agg_func == "mean":
                self.agg_func = np.mean
            else:
                raise ValueError(f"agg_func: ({agg_func}) not recognised.")
        else:
            self.agg_func = agg_func

    def reset(self):
        self.rewards.clear()
        self.errors.clear()
        self.reward_buffer.clear()
        self.error_buffer.clear()

    def record(self, reward: float, error: float) -> None:
        self.reward_buffer.append(reward)
        self.error_buffer.append(error)

    def collate(self):
        self.rewards.append(self.agg_func(self.reward_buffer))
        self.errors.append(self.agg_func(self.error_buffer))
        self.reward_buffer.clear()
        self.error_buffer.clear()

    def calculate_fitness(self) -> float:
        if self.fitness_type == "reward":
            return self.agg_func(self.rewards)
        elif self.fitness_type == "error":
            return self.agg_func(self.errors)
        else:
            return 0.0