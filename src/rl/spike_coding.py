"""
2025-08-15
"""
import numpy as np



class SpikeCoder:
    """
    An interface between RL Environment and SNN.

    Encode state / observation values to input spikes.  
    Decoder output spikes to action values.
    """
    def __init__(self, num_states: int, num_actions: int, *, input_delay: int = 3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.input_delay = input_delay
        self._delay_count = 0
        self._ready = False # Determines whether environment should be incremented

    def reset(self):
        self._delay_count = 0
        self._ready = False

    def encode(self, state: int) -> np.ndarray:
        """
        Encode the given state into spikes.

        Args:
            state (int): The state to encode.

        Returns:
            np.ndarray: A binary array representing the encoded spikes.
        """
        spikes = np.zeros(self.num_states, dtype=np.int8)
        self._ready = self._delay_count >= (self.input_delay - 1)
        if self._ready:
            spikes[state] = 1
            self._delay_count = 0
        else:
            self._delay_count += 1
        return spikes
    
    def decode(self, spikes: np.ndarray) -> int | None:
        """
        Decode the given spikes into an action.

        Args:
            spikes (np.ndarray): A binary array representing the spikes.

        Returns:
            int: The decoded action.

            Returns None only when Spike Coder is not ready (i.e. it is within waiting interval of input encoding).
        """
        if not self._ready:
            return None
        spk_idx = spikes.nonzero()[0]
        if len(spk_idx) == 0:
            action = -1
        elif len(spk_idx) > 1:
            action = np.random.choice(spk_idx).item()
        else:
            action = spk_idx.item()
        return action
    
    @property
    def ready(self) -> bool:
        """
        Determines whether environment should be incremented
        """
        return self._ready