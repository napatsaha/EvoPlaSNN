import numpy as np
from typing import Literal, Protocol, override


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
    def __init__(self, buffer_size: int, neuron_size: int):
        super().__init__()
        self.buffer_size = buffer_size
        self.neuron_size = neuron_size
        self.buffer = np.zeros((self.neuron_size, self.buffer_size), dtype=np.int_)
        self.reward_buffer = []
        self.count = 0
        self._full = False
        self.reward_null = 0.0

    def reset(self) -> None:
        """
        Reset the decoder buffer and count.
        """
        self.buffer.fill(0)
        self._full = False
        self.reward_buffer.clear()
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
        if len(np.unique(a)) == 1:
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
            reward = 1.0 if pred == label else -1.0

        self.reward_buffer.append(reward)
        return reward
    
    def get_fitness(self) -> float:
        """
        Calculate the fitness based on accumulated rewards.
        """
        if not self.reward_buffer:
            raise ValueError("No rewards recorded. Cannot calculate fitness.")
        return np.mean(self.reward_buffer)


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

        # Finally, convert to opposite
        if self.convert_type == "negate":
            return -ttfs
        elif self.convert_type == "invert":
            return np.where(ttfs == 0, 0, 1 / ttfs)
        else:
            raise ValueError("Invalid convert type. Must be 'negate' or 'invert'.")



def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def cross_entropy(p, q):
    return np.sum(p * np.log(q + 1e-10))

def mean_square_error(p, q):
    return np.sum((p - q) ** 2)