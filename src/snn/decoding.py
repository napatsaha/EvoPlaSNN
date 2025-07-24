import numpy as np
from typing import List, Literal, Protocol, Tuple, override, Union

from common.utils import make_target_times, make_target_spikes, check_max_ties
from common import math

decoder_dict = {
    "final": "FinalStepDecoder",
    "final_step": "FinalStepDecoder",
    "rate": "RateDecoder",
    "latency": "LatencyDecoder"
}

fitnessor_dict = {
    "reward": "MeanRewardFitnessor",
    "accuracy": "AccuracyFitnessor",
    "cross_entropy": "CrossEntropyFitnessor",
    "mse": "MSEFitnessor"
}

def get_decoder_class(decoder_type: str):
    """
    Get the decoder class based on the decoder type.
    """
    if decoder_type not in decoder_dict:
        raise ValueError(f"Decoder type '{decoder_type}' is not supported. Supported types: {list(decoder_dict.keys())}")
    return globals()[decoder_dict[decoder_type]]

def get_fitnessor_class(fitnessor_type: str):
    """
    Get the fitnessor class based on the fitnessor type.
    """
    if fitnessor_type not in fitnessor_dict:
        raise ValueError(f"Fitnessor type '{fitnessor_type}' is not supported. Supported types: {list(fitnessor_dict.keys())}")
    return globals()[fitnessor_dict[fitnessor_type]]

# class RewardManager:
#     def __init__(self):
#         self.memory = {"t": [], "label": [], "prediction": [], "reward": []}

#     def add(self, t: int, label: int, prediction: int, reward: float):
#         self.memory["t"].append(t)
#         self.memory["label"].append(label)
#         self.memory["prediction"].append(prediction)
#         self.memory["reward"].append(reward)

#     def calculate_reward(self, label: int, spike_out: np.ndarray, timestep: int) -> float:
#         """
#         Calculate the reward based on the label and prediction.
#         """
#         # If no spikes
#         if np.all(spike_out == 0):
#             pred = None
#             reward = 0.0
#         else:
#             pred = np.argmax(spike_out).item()
#             reward = float(np.equal(label, pred).item())

#         self.add(timestep, label, pred, reward)
#         return reward

#     def accuracy(self) -> float:
#         """
#         Calculate the accuracy of the predictions.
#         """
#         if len(self.memory["label"]) == 0:
#             return 0.0
#         accuracy = np.mean(np.equal(self.memory["label"], self.memory["prediction"]))
#         return accuracy

#     def reset(self):
#         """
#         Reset the reward manager.
#         """
#         self.memory = {"t": [], "label": [], "prediction": [], "reward": []}



class Decoder(Protocol):
    def record(self, spikes: np.ndarray) -> None:
        pass

    def decode(self, return_raw: bool = False) -> Union[int | None, Tuple[int, np.ndarray]]:
        pass

    def reset(self) -> None:
        pass

class Fitnessor(Protocol):
    def reset(self) -> None:
        """
        Reset the reward manager buffers.
        """
        pass

    def record(self, **kwargs) -> None:
        """
        Record the label, predicted output, and reward.
        """
        pass

    def calculate_fitness(self) -> float:
        """
        Get the aggregate fitness for this trial.
        """
        pass

class BaseFitnessor(Fitnessor):
    """
    Deals with combining correct label of supervisor with predicted output (e.g. from BaseDecoder) to calculate a reward.
    Also tracks fitness of the individual throughout the simulation (fitness may or may not be the same as reward).
    """
    minimise: bool = None
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reward_buffer = []
        self.output_buffer = []
        self.label_buffer = []

    def reset(self) -> None:
        """
        Reset the reward manager buffers.
        """
        self.reward_buffer.clear()
        self.output_buffer.clear()
        self.label_buffer.clear()
    
    def record(self, label, output, reward):
        """
        Record the label, predicted output, and reward.
        """
        self.reward_buffer.append(reward)
        self.label_buffer.append(label)
        self.output_buffer.append(output)

    @override
    def calculate_fitness(self, return_array: bool = False) -> float:
        """
        Calculate fitness for entire trial.
        """
        outputs, labels = self._prepare_buffers()
        fitnesses = self._calculate_fitness(outputs, labels)
        if return_array:
            return fitnesses
        else:
            return self._aggregate(fitnesses)

    def get_intermediate_fitness(self) -> List[float]:
        """
        Get the intermediate fitness values for the current trial.
        This is useful for tracking progress during the simulation.
        """
        outputs, labels = self._prepare_buffers()
        return self._calculate_fitness(outputs, labels, return_array=True)

    def _prepare_buffers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concanatenate the output and label buffers each into numpy arrays.
        """
        output_array = np.stack(self.output_buffer, axis=1)
        label_array = make_target_spikes(self.label_buffer, self.num_classes)
        return output_array, label_array
    
    def _calculate_fitness(self, outputs: np.ndarray, labels: np.ndarray, return_array: bool = False) -> float | np.ndarray:
        """
        Calculate fitness based on the outputs and labels.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _aggregate(self, values: np.ndarray) -> float:
        """
        Aggregate the fitness values.
        Depending on subclass, this may sum or average the values.
        """
        return np.sum(values)


class MeanRewardFitnessor(BaseFitnessor):
    """
    A fitnessor that calculates the mean reward from the recorded rewards.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.minimise = False  # Mean reward is typically maximised

    def _calculate_fitness(self, outputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate the mean reward from the recorded rewards.
        """
        return np.array(self.reward_buffer)
        
    def _aggregate(self, values: np.ndarray) -> float:
        """
        Aggregate the mean reward values by taking the mean.
        """
        return np.mean(values)


class AccuracyFitnessor(BaseFitnessor):
    """
    A fitnessor that calculates the accuracy of the predictions.
    Predicted outputs with more than one maximum values are treated as ties, and hence considered as incorrect.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.minimise = False  # Accuracy is typically maximised

    def _calculate_fitness(self, outputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ties = check_max_ties(outputs, axis=0)
        pred = np.argmax(outputs, axis=0)
        label = np.argmax(labels, axis=0)
        correct = np.equal(pred, label, where=~ties, out=np.zeros_like(pred))
        return correct

    def _aggregate(self, values) -> float:
        """
        Aggregate the accuracy values by taking the mean.
        """
        return np.mean(values)
    

class CrossEntropyFitnessor(BaseFitnessor):
    """
    A fitnessor that calculates the cross-entropy loss between the predicted outputs and the target labels.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.minimise = True  # Cross-entropy loss is typically minimised

    def _calculate_fitness(self, outputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate the cross-entropy loss between the predicted outputs and the target labels.
        """
        # Normalize outputs to probabilities
        output_probs = math.softmax(outputs)
        losses = math.cross_entropy_loss(labels, output_probs, axis=0)
        return losses


class MSEFitnessor(BaseFitnessor):
    """
    A fitnessor that calculates the mean squared error (MSE) between the predicted outputs and the target labels.
    Target labels differ based on whether a rate decoder or latency decoder is used.
    For latency
    """
    def __init__(self, num_classes: int, decoder_type: Literal["rate", "latency"] = "rate",
                 buffer_length: int = 10, max_value: float = None):
        super().__init__(num_classes)
        self.minimise = True  # MSE is typically minimised
        self.decoder_type = decoder_type
        self.buffer_length = buffer_length
        self.max_value = max_value if max_value is not None else -buffer_length

    @override
    def _prepare_buffers(self):
        outputs = np.stack(self.output_buffer, axis=1)
        if self.decoder_type == "rate":
            labels = make_target_spikes(self.label_buffer, self.num_classes)
        elif self.decoder_type == "latency":
            labels = make_target_times(self.label_buffer, self.num_classes, self.buffer_length, self.max_value)
        else:
            raise ValueError(f"Decoder type '{self.decoder_type}' is not supported for MSEFitnessor.")

        return outputs, labels

    def _calculate_fitness(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the mean squared error between the predicted outputs and the target labels.
        """
        mse = math.mean_square_error(labels, outputs, axis=0)
        return mse


#######
#  Decoding Methods ##


class BaseDecoder(Decoder):
    """
    Deals with converting a spike train from each output neuron into a scalar value.
    """
    def __init__(self, buffer_size: int, neuron_size: int, #fitness_type: Literal["reward", "mse", "cross_entropy"] = "reward",
                 reward_null: float = 0.0, reward_correct: float = 1.0, reward_incorrect: float = -1.0):
        super().__init__()
        # if fitness_type not in ["reward", "mse", "cross_entropy"]:
        #     raise ValueError("fitness_type must be one of ['reward', 'mse', 'cross_entropy']")
        self.buffer_size = buffer_size
        self.neuron_size = neuron_size
        self.buffer = np.zeros((self.neuron_size, self.buffer_size), dtype=np.int_)
        # self.fitness_buffer = []
        # self.fitness_type = fitness_type
        self.count = 0
        self._full = False
        self.reward_null = reward_null
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect

    def reset(self) -> None:
        """
        Reset the decoder buffer and count.
        """
        self.buffer.fill(0)
        self._full = False
        # self.fitness_buffer.clear()
        self.count = 0

    def record(self, spikes: np.ndarray) -> None:
        """
        Record spikes in the buffer.
        """
        if spikes.ndim != 1 or spikes.shape[0] != self.neuron_size:
            raise ValueError(f"Expected spikes shape ({self.neuron_size},), got {spikes.shape}")
        
        self.buffer[:, self.count] = spikes
        self.count += 1
        if self.count == self.buffer_size:
            self._full = True
        else:
            self._full = False
        self.count %= self.buffer_size

    # def decode(self, return_raw: bool = False) -> Union[int | None, Tuple[int, np.ndarray]]:
    #     """
    #     Decode the buffer into predicted class. 
    #     If tied, return None.
    #     """
    #     if not self._full:
    #         raise ValueError("Buffer is not full. Cannot decode.")
        
    #     a = self._decode()
    #     pred = self.predict(a)

    #     if return_raw:
    #         return pred, a
    #     else:
    #         return pred

    def decode(self) -> np.ndarray:
        if not self._full:
            raise ValueError("Buffer is not full. Cannot decode.")
        
        return self._decode()

    def predict(self, output):
        # Check if there is a tie
        max_value = np.max(output)
        if np.sum(output == max_value) > 1:
            pred = None
        # Otherwise return neuron with largest value
        else:
            pred = np.argmax(output)
        return pred

    def _decode(self) -> np.ndarray:
        """
        Converts the buffer into a scalar for each output neuron.
        Subclass must implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def calculate_reward(self, label: int, output: np.ndarray) -> float:
        """
        Calculate the reward based on the decoded output and given target label.
        """
        pred = self.predict(output)

        reward = self._reward_func(label, pred)

        # fitness = self.calculate_fitness(label, reward)
        # self.fitness_buffer.append(fitness)
        return reward

    def _reward_func(self, label, pred):
        if pred is None:
            reward = self.reward_null
        else:
            reward = self.reward_correct if pred == label else self.reward_incorrect
        return reward
    
    # def calculate_fitness(self, label: int, reward: float) -> float:
    #     """
    #     Calculate fitness for each example.
    #     """
    #     if self.fitness_type == "reward":
    #         return reward
    #     else:
    #         raise NotImplementedError(f"Fitness type '{self.fitness_type}' is not implemented yet.")

    # def get_fitness(self) -> float:
    #     """
    #     Calculate aggregate fitness for this trial.
    #     """
    #     if len(self.fitness_buffer) == 0:
    #         raise ValueError("No fitness recorded. Cannot calculate fitness.")
    #     return np.mean(self.fitness_buffer)


class FinalStepDecoder(BaseDecoder):
    """
    A Decoder that only uses information about the final step of a sequence.
    """
    def __init__(self, buffer_size, neuron_size, **kwargs):
        super().__init__(buffer_size, neuron_size, **kwargs)
    
    @override
    def _decode(self) -> np.ndarray:
        """
        Returns the last recorded spike train.
        """
        return self.buffer[:, -1]


class RateDecoder(BaseDecoder):
    def __init__(self, buffer_size, neuron_size, **kwargs):
        super().__init__(buffer_size, neuron_size, **kwargs)

    @override
    def _decode(self) -> np.ndarray:
        """
        Count number of spikes in each neuron.
        """
        return np.sum(self.buffer, axis=1)
    

class LatencyDecoder(BaseDecoder):
    """
    A decoder that computes the latency (time-to-first-spike) for each neuron 
    in a spiking neural network (SNN) and applies a specified transformation 
    to the result.

    Attributes:
        direction (Literal["first", "last"]): 
            Determines which direction time-to-first-spike (TTFS) is calculated from.
            If "first" (default), find the neuron that spikes first.
            If "last", find the neuron that spikes most recently.
        convert_type (Literal["negate", "invert"]): Specifies the transformation 
            to apply to the time-to-first-spike (TTFS) values. "negate" returns 
            the negative of TTFS, while "invert" returns the reciprocal of TTFS 
            (with zeros handled to avoid division by zero).

    Methods:
        _decode() -> np.ndarray:
            Decodes the buffer to compute the latency for each neuron and 
            applies the specified transformation.

    Raises:
        ValueError: If an invalid value is provided for `direction` or 
            `convert_type`.

    Parameters:
        buffer_size (int): The size of the buffer used for decoding.
        neuron_size (int): The number of neurons in the network.
        direction (Literal["first", "last"], optional): The direction to process 
            the buffer. Defaults to "first".
        convert_type (Literal["negate", "invert"], optional): The transformation 
            to apply to the TTFS values. Defaults to "negate".
    """
    def __init__(self, buffer_size, neuron_size, direction: Literal["first", "last"] = "first",
                 convert_type: Literal["negate", "invert"] = "negate", 
                 max_value: int = None, **kwargs):
        super().__init__(buffer_size, neuron_size, **kwargs)
        if direction not in ["first", "last"]:
            raise ValueError("Direction must be 'first' or 'last'.")
        if convert_type not in ["negate", "invert"]:
            raise ValueError("Convert type must be 'negate' or 'invert'.")
        self.direction = direction
        self.convert_type = convert_type
        self.max_value = max_value if max_value is not None else self.buffer_size

    @override
    def _decode(self) -> np.ndarray:
        if self.direction == "last":
            a = np.flip(self.buffer, axis=1)
        else:
            a = self.buffer
        # Find time to first spike for each neuron
        ttfs = np.argmax(a, axis=1)
        # To handle neurons that never spike, we set their time to first spike to infinity
        ttfs = np.where(np.sum(a, axis=1) == 0, self.max_value, ttfs)

        # Finally, convert to opposite
        if self.convert_type == "negate":
            return -ttfs
        elif self.convert_type == "invert":
            return math.safe_divide(1.0, ttfs, default_value=np.inf)
        else:
            raise ValueError("Invalid convert type. Must be 'negate' or 'invert'.")




