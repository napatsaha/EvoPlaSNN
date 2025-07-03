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

def fitness_func1(error, a=3, b=2):
    return a * np.exp(-error) - b

def fitness_func2(error, a=3, b=2):
    return a / (2 - np.exp(-error)) - b

def fitness_func3(error, a=3, b=2):
    return a / (1 + error) - b

fitness_func_dict = {
    "func1": fitness_func1,
    "func2": fitness_func2,
    "func3": fitness_func3
}

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
        # Reward = -1 for Error > 0, etc
        reward = 1.0 if error == 0 else -1.0
        return error, reward
    

class SimpleCollector(CollectorProtocol):
    def __init__(self, buffer_size: int = None, fitness_type: Literal["reward", "error", "mapped"] = "reward", 
                 agg_func: Callable | str = "sum", map_func: Callable | Literal["func1", "func2", "func3"] = "func3",
                 map_func_kwargs: dict = None,
                 **kwargs):
        """
        Simple collector for aggregating rewards and errors over a buffer size.
        Args:
            buffer_size (int): Maximum Size of the buffer to store rewards and errors.
            fitness_type (str): Type of fitness to calculate. Options are "reward", "error", or "mapped".
                - "reward": Aggregate rewards.
                - "error": Aggregate errors.
                - "mapped": Summed errors mapped to rewards using a mapping function.
            agg_func (Callable | str): Function to aggregate rewards/errors within-sample. Can be "sum", "mean", or a custom function.
                Only applied to "reward" or "error".
            map_func (Callable, optional): Function to map errors to rewards. Defaults to None.
                Only applied "mapped" fitness.
        """
        self.fitnesses = []
        self.rewards = []
        self.errors = []
        self._in_sample_rewards = deque(maxlen=buffer_size)
        self._in_sample_errors = deque(maxlen=buffer_size)
        if fitness_type not in ["reward", "error", "mapped"]:
            raise ValueError("fitness_type must be either 'reward', 'error' or 'mapped'.")
        self.fitness_type = fitness_type
        self._minimise = True if fitness_type == "error" else False
        # Function for mapping errors to rewards
        if isinstance(map_func, str):
            if map_func in fitness_func_dict:
                self.map_func = fitness_func_dict[map_func]
            else:
                raise ValueError(f"map_func: ({map_func}) not recognised. Available options are {list(fitness_func_dict.keys())}.")
        elif isinstance(map_func, Callable):
            self.map_func = map_func
        else:
            raise ValueError("map_func must be either a string or a callable function.")
        # Custom mapping function arguments
        if map_func_kwargs is not None:
            self.map_func_kwargs = map_func_kwargs
        else:
            self.map_func_kwargs = {}
        # if map_func is not None:
        #     if not callable(map_func):
        #         raise ValueError("map_func must be a callable function.")
        #     self.map_func = map_func
        # elif map_func is None:
        #     self.map_func = fitness_func
        # Function for aggregating
        if isinstance(agg_func, str):
            if agg_func == "sum":
                self.agg_func = sum
            elif agg_func == "mean":
                self.agg_func = np.mean
            else:
                raise ValueError(f"agg_func: ({agg_func}) not recognised.")
        elif isinstance(agg_func, Callable):
            self.agg_func = agg_func
        else:
            raise ValueError("agg_func must be either a string name or a callable function.")

    def reset(self):
        self.fitnesses.clear()
        self.rewards.clear()
        self.errors.clear()
        self._in_sample_rewards.clear()
        self._in_sample_errors.clear()

    def record(self, reward: float, error: float) -> None:
        self._in_sample_rewards.append(int(reward))
        self._in_sample_errors.append(int(error))

    def collate(self):
        # Map total errors to reward using designated mapping function
        total_errors = np.sum(self._in_sample_errors)
        fitness = self.map_func(total_errors, **self.map_func_kwargs)
        self.fitnesses.append(float(fitness))
        # Append the aggregated rewards and errors to the lists
        self.rewards.append(float(self.agg_func(self._in_sample_rewards)))
        self.errors.append(float(self.agg_func(self._in_sample_errors)))
        # self.rewards.append(self.agg_func(self.reward_buffer))
        # self.errors.append(self.agg_func(self.error_buffer))

        # Clear existing buffer for this sample
        self._in_sample_rewards.clear()
        self._in_sample_errors.clear()

    def calculate_fitness(self) -> float:
        if self.fitness_type == "reward":
            return np.mean(self.rewards)
        elif self.fitness_type == "error":
            return np.mean(self.errors)
        elif self.fitness_type == "mapped":
            return np.mean(self.fitnesses)
        else:
            return 0.0