import numpy as np
from typing import Literal, Protocol, override


decoder_dict = {
    "final": "FinalStepDecoder",
    "final_step": "FinalStepDecoder",
    "rate": "RateDecoder",
    "latency": "LatencyDecoder"
}

def get_decoder_class(decoder_type: str):
    """
    Get the decoder class based on the decoder type.
    """
    if decoder_type not in decoder_dict:
        raise ValueError(f"Decoder type '{decoder_type}' is not supported. Supported types: {list(decoder_dict.keys())}")
    return globals()[decoder_dict[decoder_type]]


class RewardManager:
    def __init__(self):
        self.memory = {"t": [], "label": [], "prediction": [], "reward": []}

    def add(self, t: int, label: int, prediction: int, reward: float):
        self.memory["t"].append(t)
        self.memory["label"].append(label)
        self.memory["prediction"].append(prediction)
        self.memory["reward"].append(reward)

    def calculate_reward(self, label: int, spike_out: np.ndarray, timestep: int) -> float:
        """
        Calculate the reward based on the label and prediction.
        """
        # If no spikes
        if np.all(spike_out == 0):
            pred = None
            reward = 0.0
        else:
            pred = np.argmax(spike_out).item()
            reward = float(np.equal(label, pred).item())

        self.add(timestep, label, pred, reward)
        return reward

    def accuracy(self) -> float:
        """
        Calculate the accuracy of the predictions.
        """
        if len(self.memory["label"]) == 0:
            return 0.0
        accuracy = np.mean(np.equal(self.memory["label"], self.memory["prediction"]))
        return accuracy

    def reset(self):
        """
        Reset the reward manager.
        """
        self.memory = {"t": [], "label": [], "prediction": [], "reward": []}


class Decoder(Protocol):
    def record(self, spikes: np.ndarray) -> None:
        pass

    def calculate_reward(self, target) -> float:
        pass

    def reset(self) -> None:
        pass


class BaseDecoder(Decoder):
    def __init__(self, buffer_size: int, neuron_size: int, fitness_type: Literal["reward", "mse", "cross_entropy"] = "reward",
                 reward_null: float = 0.0, reward_correct: float = 1.0, reward_incorrect: float = -1.0):
        super().__init__()
        if fitness_type not in ["reward", "mse", "cross_entropy"]:
            raise ValueError("fitness_type must be one of ['reward', 'mse', 'cross_entropy']")
        self.buffer_size = buffer_size
        self.neuron_size = neuron_size
        self.buffer = np.zeros((self.neuron_size, self.buffer_size), dtype=np.int_)
        self.fitness_buffer = []
        self.fitness_type = fitness_type
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
        self.fitness_buffer.clear()
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

    def decode(self) -> int | None:
        """
        Decode the buffer into predicted class. 
        If tied, return None.
        """
        if not self._full:
            raise ValueError("Buffer is not full. Cannot decode.")
        
        a = self._decode()
        # Check if there is a tie
        max_value = np.max(a)
        if np.sum(a == max_value) > 1:
            return None
        # Otherwise return neuron with largest value
        else:
            return np.argmax(a)

    def _decode(self) -> np.ndarray:
        """
        Converts the buffer into a scalar for each output neuron.
        Subclass must implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def calculate_reward(self, label: int) -> float:
        """
        Calculate the reward based on the decoded output and given target label.
        """
        pred = self.decode()
        if pred is None:
            reward = self.reward_null
        else:
            reward = self.reward_correct if pred == label else self.reward_incorrect

        fitness = self.calculate_fitness(label, reward)
        self.fitness_buffer.append(fitness)
        return reward
    
    def calculate_fitness(self, label: int, reward: float) -> float:
        """
        Calculate fitness for each example.
        """
        if self.fitness_type == "reward":
            return reward
        else:
            raise NotImplementedError(f"Fitness type '{self.fitness_type}' is not implemented yet.")



    def get_fitness(self) -> float:
        """
        Calculate aggregate fitness for this trial.
        """
        if len(self.fitness_buffer) == 0:
            raise ValueError("No fitness recorded. Cannot calculate fitness.")
        return np.mean(self.fitness_buffer)


class FinalStepDecoder(BaseDecoder):
    """
    A Decoder that only uses information about the final step of a sequence.
    """
    def __init__(self, buffer_size, neuron_size):
        super().__init__(buffer_size, neuron_size)
    
    @override
    def _decode(self) -> np.ndarray:
        """
        Returns the last recorded spike train.
        """
        return self.buffer[:, -1]


class RateDecoder(BaseDecoder):
    def __init__(self, buffer_size, neuron_size):
        super().__init__(buffer_size, neuron_size)

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
                 convert_type: Literal["negate", "invert"] = "negate"):
        super().__init__(buffer_size, neuron_size)
        if direction not in ["first", "last"]:
            raise ValueError("Direction must be 'first' or 'last'.")
        if convert_type not in ["negate", "invert"]:
            raise ValueError("Convert type must be 'negate' or 'invert'.")
        self.direction = direction
        self.convert_type = convert_type

    @override
    def _decode(self) -> np.ndarray:
        if self.direction == "last":
            a = np.flip(self.buffer, axis=1)
        else:
            a = self.buffer
        # Find time to first spike for each neuron
        ttfs = np.argmax(a, axis=1)
        # To handle neurons that never spike, we set their time to first spike to infinity
        ttfs = np.where(np.sum(a, axis=1) == 0, np.inf, ttfs)

        # Finally, convert to opposite
        if self.convert_type == "negate":
            return -ttfs
        elif self.convert_type == "invert":
            return safe_divide(1.0, ttfs, default_value=np.inf)
        else:
            raise ValueError("Invalid convert type. Must be 'negate' or 'invert'.")



def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def cross_entropy(p, q):
    return np.sum(p * np.log(q + 1e-10))

def mean_square_error(p, q):
    return np.sum((p - q) ** 2)

def safe_divide(x, y, default_value=np.inf):
    """Safe division that avoids division by zero."""
    return np.divide(x, y, where=y != 0, out=np.full(np.broadcast(x, y).shape, default_value, dtype=float))