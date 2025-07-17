import numpy as np
from typing import Literal, Protocol, Tuple, Callable
from collections import deque

from snn.spikegen import BinaryArrayGenerator

def create_rewarder(type: str, **kwargs):
    if type == "simple":
        class_name = "SimpleRewarder"
    elif type == "weighted":
        class_name = "WeightedRewarder"
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
    def __init__(self, num_classes, pattern_length: int, spacing: int = 0, interval: int = None, *,
                 target_position: Literal["first", "last", "rate"] = "last", **kwargs):
        self.num_classes = num_classes
        # self.spikegen = spikegen
        self.pattern_length = pattern_length
        self.spacing = spacing
        self.interval = interval if interval is not None else 1
        self.full_length = self.pattern_length + self.spacing

        self.target_position = target_position
        if self.target_position == "first":
            self._label = 0
        elif self.target_position == "last":
            self._label = self.pattern_length - 1
        elif self.target_position == "rate":
            self._label = np.arange(0, self.pattern_length, self.interval)
        
        # self._label = 0 if target_position == "first" else self.pattern_length - 1 if target_position == "last" else None
        self._create_target_array()
        self.reset()

    def _create_target_array(self):
        # Create pre-generated array of target outputs for easier slicing
        self.target_array = np.zeros((self.num_classes, self.num_classes, self.pattern_length), dtype=np.int_)
        for i in range(self.num_classes):
            self.target_array[i, i, self._label] = 1
            # if self.target_position == "rate":
            #     if self.interval is None:
            #         self.target_array[i, i, :] = 1
            #     else:
            #         idx = np.arange(0, self.pattern_length, self.interval)
            #         self.target_array[i, i, idx] = 1
            # else:
            #     self.target_array[i, i, self._label] = 1
        if self.spacing > 0:
            # Add spacing after pattern
            self.target_array = np.pad(self.target_array, ((0, 0), (0, 0), (0, self.spacing)), mode='constant', constant_values=0)
            # self.full_length = self.target_array.shape[2] - 1

    def find_active_indices(self, input_array: np.ndarray) -> np.ndarray:
        # Just make it only work with 2D array
        # if input_array.ndim == 2:
        active = (np.max(input_array, axis=0) > 0).astype(np.int8)
        # Find the indices of the active spikes based on the target position
        if self.target_position == "last":
            idx = self.full_length - np.argmax(np.flip(active, axis=-1), axis=-1) - 1
        elif self.target_position == "first":
            idx = np.argmax(active, axis=-1)
        elif self.target_position == "rate":
            idx = np.nonzero(active)
        # elif input_array.ndim == 3:
        #     active = (np.max(input_array, axis=-2) > 0).astype(np.int8)
        #     # Find the indices of the active spikes based on the target position
        #     if self.target_position == "last":
        #         idx = -np.argmax(np.flip(active, axis=-1), axis=-1)
        #     elif self.target_position == "first":
        #         idx = np.argmax(active, axis=-1)
        #     elif self.target_position == "rate":
        #         idx = np.nonzero(active)
        # np.put_along_axis(self.target_array, idx[:, np.newaxis], 1, axis=1)
        return idx

    def update_target_array(self, input_array: np.ndarray, current_class: int = None) -> None:
        """
        Update the target array based on the input array.
        This is useful if the target array needs to be dynamically generated or updated.
        """        
        if current_class is None and input_array.ndim == 3:
            # Bulk update for all classes
            self.target_array.fill(0)  # Reset the target array
            for i in range(self.num_classes):
                idx = self.find_active_indices(input_array[i])
                self.target_array[i, i, idx] = 1
        elif current_class is not None and input_array.ndim == 2:
            if current_class < 0 or current_class >= self.num_classes:
                raise ValueError(f"current_class must be between 0 and {self.num_classes - 1}. Got {current_class}.")
            self.target_array[current_class].fill(0)  # Reset the target array for the current class
            idx = self.find_active_indices(input_array)
            self.target_array[current_class, current_class, idx] = 1
        else:
            raise ValueError("Input array must be 2D for single class update or 3D for bulk update.")

    def reset(self):
        self.count = 0

    def get_target(self, current_class):
        idx = self.count % self.full_length

        target_spikes = self.target_array[current_class, :, idx]
        if self.count >= self.full_length:
            self.count = 0
        
        self.count += 1
        return target_spikes
    
    def get_reward(self, current_class, output_spikes):
        target_spikes = self.get_target(current_class)
        error, reward = self._calculate_error_and_reward(target_spikes, output_spikes)
        return error, reward

    def _calculate_error_and_reward(self, target_spikes, output_spikes):
        # Sum the absolute differences between target and output spikes
        error = np.sum(np.abs(target_spikes - output_spikes))
        # Reward = +1 for Error = 0,
        # Reward = -1 for Error > 0, etc
        reward = 1.0 if error == 0 else -1.0
        return error,reward
    
    def get_max_reward(self):
        """
        Get the maximum possible reward for the current target array.
        This is simply the number of classes, since each class has a target spike at one position.
        """
        return 1.0


class WeightedRewarder_old(SimpleRewarder):
    """
    Weighted Rewarder version that only works with Array-based Spike Generator.
    """
    def __init__(self, num_classes, spikegen: BinaryArrayGenerator, *,
                 ignore_silent_inputs: bool = True,
                 log_scale: bool = False,
                 target_position: Literal["first", "last"] = "last", **kwargs):
        super().__init__(num_classes, spikegen.pattern_length, spikegen.spacing,
                         target_position=target_position, **kwargs)
        self.ignore_silent_inputs = ignore_silent_inputs
        self.log_scale = log_scale
        self.weights = self.create_weights(spikegen)

    def create_weights(self, spikegen: BinaryArrayGenerator):
        # Just an easier way to write self.target_array
        A = self.target_array
        # Create mask to block out non-input timesteps (i.e. during intervals)
        mask = np.sum(spikegen.array, axis=(1,))
        mask = np.logical_not(mask.astype(np.bool))
        mask = np.broadcast_to(mask, A.shape)
        A_masked = np.ma.masked_array(A, mask)
        # number of non-silent timesteps
        if self.ignore_silent_inputs:
            LENG = (spikegen.pattern_length + spikegen.interval - 1) / spikegen.interval
        else:
            LENG = spikegen.pattern_length
        # total number of spikes in each class (assuming every class has same number of total spikes)
        CNT = np.max(A_masked, axis=(1)).sum(axis=1).data[0]
        RATE = CNT / LENG
        # If silent, Weight = 1 / (1 - RATE) <- Lower  (e.g. 1/0.96 = 1.04)
        # If active, Weight = 1 / RATE       <- Higher (e.g. 1/0.04 = 25)
        wts = 1 / np.abs(A_masked.max(axis=1) - 1 + RATE)
        # Apply log scaling if required
        if self.log_scale:
            wts = np.log(wts)
        # Block out rewards during interval timesteps
        wts = np.where(wts.mask, 0.0, wts)
        return wts

    def get_max_reward(self):
        """
        Get the maximum possible reward for the current target array.
        This is simply the sum of the weights averaged between class.
        """
        return np.sum(self.weights, axis=1).mean()

    def get_target(self, current_class):
        idx = self.count % self.full_length

        target_spikes = self.target_array[current_class, :, idx]
        wts = self.weights[current_class, idx]
        if self.count >= self.full_length:
            self.count = 0
        
        self.count += 1
        return target_spikes, wts

    def get_reward(self, target_spikes, output_spikes):
        target_spikes, wts = target_spikes
        error, reward = super().get_reward(target_spikes, output_spikes)
        # Apply weights to the reward
        reward *= wts
        return error, reward


class WeightedRewarder(SimpleRewarder):
    def __init__(self, num_classes, pattern_length: int, spacing: int = 0, interval: int = None, *,
                 ignore_silent_inputs: bool = False,
                 log_scale: bool = False,
                 target_position: Literal["first", "last", "rate"] = "last", **kwargs):
        super().__init__(num_classes, pattern_length, spacing, interval,
                         target_position=target_position, **kwargs)
        self.ignore_silent_inputs = ignore_silent_inputs
        self.log_scale = log_scale
        self.min_denom = 1e-3
        # self.weights = self.create_weights()
        if self.interval > 1 and self.ignore_silent_inputs:
            self._spk_leng = (self.pattern_length + self.interval - 1) / self.interval
        else:
            self._spk_leng = self.pattern_length
        self._update_rate()
        # self._create_weights()

    def _update_rate(self):
        """
        Update spike rates to be used for weighted rewards, based on internal target array.
        Can be called after updating target array.
        """
        self._spk_cnt = np.max(self.target_array, axis=(1,)).sum(axis=1)
        self._spk_rate = self._spk_cnt / self._spk_leng
        # Prevent division by zero when all spikes are silent or all spikes are active
        self.pos_wt = 1 / np.where(self._spk_rate == 0, self.min_denom, self._spk_rate)
        self.neg_wt = 1 / np.where(self._spk_rate == 1, self.min_denom, (1 - self._spk_rate))

    def _create_weights(self):
        active = np.max(self.target_array, axis=(1,))
        self.weights = np.zeros_like(active, dtype=np.float32)
        self.weights = np.where(active > 0, np.broadcast_to(self.pos_wts, active.T.shape).T, 
                                np.broadcast_to(self.neg_wts, active.T.shape).T)
        # Apply log scaling if required
        if self.log_scale:
            self.weights = np.log(self.weights)

    # def create_weights(self):
    #     """
    #     New version, without relying on spikegen.array
    #     """
    #     # Simply tells if any output spike occurs at each timestep or not
    #     active = (np.max(self.target_array, axis=(1,)) > 0).astype(np.int8)
    #     # number of non-silent timesteps
    #     if self.interval > 1 and self.ignore_silent_inputs:
    #         leng = (self.pattern_length + self.interval - 1) / self.interval
    #     else:
    #         leng = self.pattern_length
    #     # total number of spikes in each class (assuming every class has same number of total spikes)
    #     cnt = active.sum(axis=1)
    #     spk_rate = cnt / leng
    #     spk_rate = np.broadcast_to(spk_rate, active.T.shape).T
    #     # If silent, Weight = 1 / (1 - RATE) <- Lower  (e.g. 1/0.96 = 1.04)
    #     # If active, Weight = 1 / RATE       <- Higher (e.g. 1/0.04 = 25)
    #     wts = 1 / np.abs(active - 1 + spk_rate)
    #     # Apply log scaling if required
    #     if self.log_scale:
    #         wts = np.log(wts)
    #     return wts.astype(np.float32)

    def update_target_array(self, input_array, current_class: int = None) -> None:
        """
        Same as parent's update_target_array, but also updates the weights.
        """
        super().update_target_array(input_array, current_class)
        # self.weights = self.create_weights()
        self._update_rate()

    # def get_max_reward(self):
    #     """
    #     Get the maximum possible reward for the current target array.
    #     This is simply the sum of the weights averaged between class.
    #     """
    #     return np.sum(self.weights, axis=1).mean()

    def get_reward(self, current_class, output_spikes):
        target_spikes = self.get_target(current_class)
        # Get Binary rewards (sum of errors)
        error, reward = self._calculate_error_and_reward(target_spikes, output_spikes)
        # Apply weights based on whether target spikes are all silent or not
        wts = self.pos_wt[current_class] if np.sum(target_spikes) > 0 else self.neg_wt[current_class]
        # Apply weights to the reward
        reward *= wts
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
        self.buffer_size = buffer_size
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
        self._in_sample_rewards.append(reward)
        self._in_sample_errors.append(error)

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