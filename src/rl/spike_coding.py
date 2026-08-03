"""
2025-08-15
"""
from common.base import SpikeCoder
import numpy as np
import gymnasium as gym


class StateCoder(SpikeCoder):
    """
    An interface between RL Environment and SNN.

    Encode state / observation values to input spikes.  
    Decoder output spikes to action values.
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, *,
                 input_delay: int = 3, output_delay: int = 0):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_states = obs_space.n # Assumed to be a Discrete space
        self.num_actions = action_space.n
        self.input_delay = input_delay
        self._delay_count = 0
        self._ready = False # Determines whether environment should be incremented
        self._apply_output_delay = output_delay > 0
        if self._apply_output_delay:
            self.output_delay = output_delay
            self._output_ready = False
            self._output_delay_count = 0


    def reset(self):
        self._delay_count = 0
        self._ready = False
        if self._apply_output_delay:
            self._output_ready = False
            self._output_delay_count = 0

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
        if self._apply_output_delay:
            self._output_ready = self._output_delay_count >= (self.input_delay + self.output_delay - 1)
            if self._output_ready:
                self._output_delay_count = self.output_delay
            else:
                self._output_delay_count += 1
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
        if self._apply_output_delay:
            if not self._output_ready:
                return None
        elif not self._ready:
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
        if self._apply_output_delay:
            return self._output_ready
        else:
            return self._ready
        
    @property
    def input_size(self) -> int:
        return self.num_states
    @property
    def output_size(self) -> int:
        return self.num_actions


class ObservationCoder(SpikeCoder):
    """
    Spike Coder which converts an observation array of real-values into spikes with the same number of channels as the observation array.

    Uses temporal encoding and decoding. 
    Encodes observation array into input neurons by times of spikes.  
    Decodes output neurons into actions based on which one spike first.
    """
    def __init__(self, obs_space: gym.spaces, action_space: gym.spaces, time_window: int = 10):
        self.obs_space = obs_space
        self.action_space = action_space
        self.time_window = time_window
        self.num_obs = obs_space.shape[0]
        self.num_actions = action_space.n

        # Container objects
        self.array = np.zeros((self.num_obs, self.time_window), dtype=np.int8)
        self.output_buffer = np.zeros((self.num_actions, self.time_window), dtype=np.int8)

        # Boolean flags
        self._ready = False # Whether or not the window has finished, and decoding can begin
        self._should_parse = True # Whether or not to process new observations into spike. (only occurs at start of time window)

        # Counting metrics
        self._count = 0

    def _parse_input(self, obs: np.ndarray):
        """
        Convert real values to spike times by scaling low-high to time window.
        """
        low, high = self.obs_space.low, self.obs_space.high
        scaled_obs = (obs - low) / (high - low)
        timing = (scaled_obs * (self.time_window - 1)).round(0).astype(int)
        return timing
    
    def encode(self, obs: np.ndarray) -> np.ndarray:
        if self._should_parse:
            self._soft_reset()
            timing = self._parse_input(obs)
            np.put_along_axis(self.array, timing[:, np.newaxis], 1, axis=1)
            self._should_parse = False

        spikes = self.array[:, self._count]
        self._count += 1
        self._ready = self._count >= (self.time_window)
        return spikes

    def decode(self, spikes: np.ndarray) -> int | None:
        self.output_buffer[:, (self._count - 1)] = spikes
        if not self._ready:
            return None
        else:
            # Find time to first spike for each neuron
            ttfs = np.argmax(self.output_buffer, axis=1)
            # To handle neurons that never spike, we set their time to first spike to the time window (maximum)
            ttfs = np.where(np.sum(self.output_buffer, axis=1) == 0, self.time_window, ttfs)
            self._should_parse = True
            return int(np.argmin(ttfs))
        
    def _soft_reset(self):
        self.output_buffer.fill(0)
        self.array.fill(0)
        self._ready = False
        self._count = 0

    def reset(self):
        self._soft_reset()
        self._should_parse = True

    @property
    def ready(self) -> bool:
        return self._ready
    
    @property
    def input_size(self) -> int:
        return self.num_obs
    @property
    def output_size(self) -> int:
        return self.num_actions
    
