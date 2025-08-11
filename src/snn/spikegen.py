
from typing import List, Literal, Tuple
import numpy as np


from snn.base import SpikeGenerator



class RandomSpikeGenerator(SpikeGenerator):
    """
    Generates random spikes based on a given distribution.
    """
    # For now, only binomial distribution is supported
    _accepted_dist = ["binomial"]

    def __init__(self, input_size: int, dist: str = "binomial", seed=None, **kwargs):
        super().__init__(input_size, seed)
        if dist not in self._accepted_dist:
            raise ValueError(f"Distribution {dist} not supported")
        self.dist = dist
        # if dist == "binomial":
        #     self.function = self.rng.binomial
        self.dist_args = kwargs

    def generate(self):
        if self.dist == "binomial":
            return self.rng.binomial(1, **self.dist_args, size=self.input_size)
        else:
            raise NotImplementedError(f"Distribution {self.dist} not implemented")


class PatternSpikeGenerator(SpikeGenerator):
    """
    Generates a pattern of sequential spikes for each neuron in the input layer.
    Delays between spikes can be specified by the interval argument.
    """
    def __init__(self, input_size: int, interval: float, *, spacing: int = None, 
                 signal_on_end: bool = False, starting_neuron: int = None, loop: bool = False,
                 ascending: bool = True, start_spike: bool = True, seed=None):
        super().__init__(input_size, seed)
        self.interval = max(1, int(interval))
        self.spacing = max(1, int(spacing)) if spacing is not None else interval
        self.signal_on_end = signal_on_end

        # Spiking behaviour
        self.ascending = ascending
        self.start_spike = start_spike
        self.starting_neuron = max(0, min(input_size - 1, starting_neuron)) if starting_neuron is not None else 0 if ascending else input_size - 1
        self.max_spike_count = self.input_size if loop else self.input_size - self.starting_neuron if self.ascending else self.starting_neuron + 1
        self.direction = 1 if ascending else -1
        self.reset()
        # self.delay_count = self.interval if start_spike else 0
        # self.spike_count = 0
        # self.current_neuron = 0 if ascending else input_size - 1
        # self.finished = False

    def reset(self):
        self.delay_count = self.interval if self.start_spike else 0
        # if self.spacing > 0:
        # self.current_neuron = 0 if self.ascending else self.input_size - 1
        self.current_neuron = self.starting_neuron
        # else:
        #     self.current_neuron = - self.spacing if self.ascending else self.input_size + self.spacing - 1
        self.finished = False
        self.spike_count = 0

    def generate(self) -> np.ndarray:
        """
        Generates a spike pattern for the input layer.
        The pattern consists of spikes that occur at regular intervals.
        """
        self.delay_count += 1
        spikes =  self.generate_empty()

        # Intermission
        if self.finished:
            if self.delay_count >= self.spacing:
                # self.delay_count += 1
                self.reset()
                # spikes =  self.generate_empty()
            # else:
            # if self.delay_count >= (self.spacing - 1):
            #     self.reset()
        # Regular Spiking
        if not self.finished:
            if self.delay_count >= self.interval:
                # Reset delay count
                self.delay_count = 0

                # Create a spike pattern
                # spikes = self.generate_empty()
                spikes[self.current_neuron] = 1

                # Update the current neuron index
                self.current_neuron = (self.current_neuron + self.direction) % self.input_size
                self.spike_count += 1
                # Check if the pattern is finished
                if self.spike_count >= self.max_spike_count:
                    self.finished = True
                    self.spike_count = 0

            # else:
            #     spikes = self.generate_empty()
            
        return spikes

    def return_signal(self) -> bool | int:
        """
        Returns a signal that indicates whether the pattern is finished.
        """
        if self.signal_on_end:
            return self.finished
        else:
            return True

    def __len__(self):
        return self.interval * (self.max_spike_count - 1) + self.spacing


class BinaryClassGenerator(SpikeGenerator):
    """
    A SpikeGenerator that alternates between two classes of PatternSpikeGenerators with opposite directions.
    """
    def __init__(self, input_size, interval: int, spacing: int = None, *, start_spike: bool = True, 
                 signal_on_end: bool = False, reflect: bool = False,
                 starting_class: Literal["ascending", "descending"] = "ascending", p: float = 0.5, seed=None):
        super().__init__(input_size, seed)
        self.interval = max(1, int(interval))
        self.spacing = max(1, int(spacing)) if spacing is not None else self.interval
        self.p = p
        self.reflect = reflect
        self.signal_on_end = signal_on_end
        if self.reflect:
            starting_neurons = [1, input_size - 2]
        else:
            starting_neurons = [None, None]
        self.generators = [
            PatternSpikeGenerator(input_size, self.interval, spacing=self.spacing, ascending=True, start_spike=start_spike, starting_neuron=starting_neurons[0], loop=False),
            PatternSpikeGenerator(input_size, self.interval, spacing=self.spacing, ascending=False, start_spike=start_spike, starting_neuron=starting_neurons[1], loop=False)
        ]
        self.starting_class = starting_class.lower()
        self._pattern_length = (self.input_size - 1) * self.interval + 1
        self._full_length = len(self.generators[0])
        self.reset()

    def reset(self):
        self._current_class = 0 if self.starting_class == "ascending" else 1
        self.count = 0
        self.switch = False
        self._finished = False
        self._ready = False

    def _generate_spikes(self):
        self.count += 1
        gen = self.generators[self._current_class]
        spikes = gen.generate()
        finished = gen.finished
        self.switch = self.count >= len(gen)
        if self.switch:
            self._switch_class()
            self.generators[self._current_class].reset()
            self.switch = False
            self.count = 0
        return spikes, finished
    
    def _switch_class(self):
        r = self.rng.random()
        if r < self.p:
            self._current_class = 1 - self._current_class
        # self.current_class = int(r < self.p)
    
    def generate(self) -> np.ndarray:
        spikes, finished = self._generate_spikes()
        self._ready = finished and not self._finished
        self._finished = finished

        return spikes
        
    def return_signal(self):
        if self.signal_on_end:
            return self._ready
        else:
            return True

    def get_label(self) -> None | int:
        if not self._ready:
            return None
        else:
            return self._current_class

    def __len__(self):
        return self._full_length
    
    @property
    def length(self):
        """
        Returns the total number of time steps in a single pattern (including spacing).
        """
        return self._full_length

    @property
    def pattern_length(self):
        """
        Returns the duration of a single pattern (disregarding spacing in-between patterns).
        """
        return self._pattern_length

    @property
    def finished(self) -> bool:
        """
        Returns whether the current pattern is finished (including last time step in pattern).
        """
        return self._finished

    @property
    def active(self) -> bool:
        """
        Returns whether the generator is currently generating spikes (excluding spacing but including last time step).
        """
        return (self.count <= self._pattern_length)
    
    @property
    def ready(self) -> bool:
        """
        Returns whether the generator is ready to allow weight updates or reward to be calculated.  
        If `signal_on_end=True`, returns True only at last time step of pattern (to facilitate `on-reward` updates).  
        Otherwise, returns True at every time step (to facilitate `on-spike` update).
        """
        return self._ready if self.signal_on_end else True


def construct_array(n, interval, ascending=True, failure_rate=0.0, jitter=0):
    # X = np.zeros((n, spacing), dtype=np.int_)
    A = np.zeros((n, (n-1)*interval + 1), dtype=np.int_)
    # if ascending:
    row_iter = range(0, n, 1)
    # else:
    #     row_iter = range(n-1, -1, -1)
    col_iter = range(0, n*interval, interval)
    for i, j in zip(row_iter, col_iter):
        # Failure rate is the probability of skipping current neuron's spike
        if failure_rate > 0:
            p = np.random.rand()
            if p < failure_rate:
                continue
        # Jitter is std in Normal distribution (dt unit), applied as perturbation to spike timing (j)
        if jitter > 0:
            t_offset = np.round(np.random.normal(0, jitter)).astype(np.int_)
            j = np.clip(j + t_offset, 0, A.shape[1] - 1)
        A[i, j] = 1
    if not ascending:
        A = np.flip(A, axis=0)
    return A


class ArrayPatternGenerator(SpikeGenerator):
    def __init__(self, input_size, interval: int = 1, spacing: int = None, ascending: bool = True,
                 *, seed=None):
        super().__init__(input_size, seed)
        self.interval = max(1, int(interval))
        self.spacing = max(1, int(spacing)) if spacing is not None else self.interval
        self.ascending = ascending
        self.array = construct_array(self.input_size, self.interval, self.ascending)
        if self.spacing > 0:
            self.array = np.pad(self.array, ((0, 0), (0, self.spacing)), mode='constant', constant_values=0)
        self._pattern_length = (self.input_size - 1) * self.interval + 1
        self._full_length = self.array.shape[1]
        self.reset()

    def reset(self):
        self.count = 0
        self.finished = False

    def generate(self) -> np.ndarray:
        idx = self.count % self._full_length
        if self.count >= self._pattern_length:
            self.finished = True
        if self.count >= self._full_length:
            self.count = 0
            self.finished = False
        spikes = self.array[:, idx]
        self.count += 1
        return spikes


class BinaryArrayGenerator(SpikeGenerator):
    class_order = ["ascending", "descending"]
    current_class: int = None
    def __init__(self, input_size, interval: int = 1, spacing: int = None, p: float = 1.0,
                 starting_class: Literal["ascending", "descending"] = None, 
                 *, seed=None):
        super().__init__(input_size, seed)
        self.num_classes = 2
        self.interval = max(1, int(interval))
        self.spacing = max(0, int(spacing)) if spacing is not None else self.interval
        self.p = min(1.0, max(0.0, p))
        self._init_array()
        self._starting_class = starting_class
        self._pattern_length = (self.input_size - 1) * self.interval + 1
        self._full_length = self.array.shape[2]
        self.reset()

    def _init_array(self):
        A = construct_array(self.input_size, self.interval, ascending=True)
        B = construct_array(self.input_size, self.interval, ascending=False)
        self.array = np.concatenate([A[np.newaxis, ...], B[np.newaxis, ...]], axis=0)
        self._pad_arrays()

    def _pad_arrays(self):
        """
        Pads each array to the full length by adding zeros at the end.
        """
        self.array = np.pad(self.array, ((0, 0), (0, 0), (0, self.spacing)), mode='constant', constant_values=0)

    def reset(self):
        if self._starting_class is None:
            self.current_class = int(self.rng.random() > self.p)
        else:
            try:
                self.current_class = self.class_order.index(self._starting_class.lower())
            except:
                raise ValueError(f"Starting class must be one of {self.class_order}")
        self.count = 0
        self._finished = False

    def switch(self):
        r = self.rng.random()
        if r < self.p:
            self.current_class = 1 - self.current_class

    def generate(self) -> np.ndarray:
        # Find index before changing count
        idx = self.count % self._full_length
        
        # When spiking pattern is finished
        if self.count >= (self._pattern_length - 1):
            self._finished = True

        # When waiting time is finished, reset to initial state
        if self.count >= self._full_length:
            self._finished = False
            self.count = 0
            self.switch()

        # Increment count after resetting state
        self.count += 1

        # Slice stored array using recently resetted class
        spikes = self.array[self.current_class, :, idx]
        return spikes

    def return_signal(self):
        return self._finished and (self.count < self._pattern_length + 1)

    # def get_label(self) -> None | int:
    #     """
    #     Returns the current class label.
    #     """
    #     if self.return_signal():
    #         return self.current_class
    #     else:
    #         return None

    # def __len__(self):
    #     return self._full_length
    
    # @property
    # def length(self):
    #     """
    #     Returns the total number of time steps in a single pattern (including spacing).
    #     """
    #     return self._full_length

    # @property
    # def pattern_length(self):
    #     """
    #     Returns the duration of a single pattern (disregarding spacing in-between patterns).
    #     """
    #     return self._pattern_length
    
    # @property
    # def finished(self) -> bool:
    #     """
    #     Returns whether the current pattern is finished (including last time step in pattern).
    #     """
    #     return self._finished
    # @property
    # def active(self):
    #     """
    #     Returns whether the pattern is being generated or whether it is during a waiting period.
    #     """
    #     return (self.count <= self._pattern_length)

    # @property
    # def ready(self) -> bool:
    #     """
    #     Returns whether the generator is ready to allow weight updates or reward to be calculated.  
    #     """
    #     return self._finished and self.active

   
class CustomArrayGenerator(SpikeGenerator):
    """
    An spike generator that accepts any arbritary number of arrays as input patterns, each corresponding to an individual class.
    """
    def __init__(self, input_size: int, arrays: List[np.ndarray], p: float = 1.0, spacing: int = None,
                 starting_class: int = None, *, seed=None):
        super().__init__(input_size, seed)
        self.p = min(1.0, max(0.0, p))
        self.num_classes = len(arrays)
        self._starting_class = starting_class
        self.array = arrays
        self._validate_arrays()
        self.spacing = max(0, int(spacing)) if spacing is not None else 0
        self._full_length = self._pattern_length + self.spacing
        if self.spacing > 0:
            self._pad_arrays()
        self.reset()

    def _validate_arrays(self):
        for array in self.array:
            array = np.asarray(array)
        ndims = [arr.ndim for arr in self.array]
        if not all(ndim == 2 for ndim in ndims):
            raise ValueError("All arrays must be 2D (shape: [num_classes, input_size])")
        input_shapes = [arr.shape[0] for arr in self.array]
        # Check if all input_sizes are the same without using self.input_size
        if len(set(input_shapes)) != 1:
            raise ValueError("All arrays must have the same input size.")
        pattern_length = [arr.shape[1] for arr in self.array]
        if len(set(pattern_length)) != 1:
            raise ValueError("All arrays must have the same pattern length.")
        self.input_size = input_shapes[0]
        self._pattern_length = pattern_length[0]
        self.array = np.stack(self.array, axis=0)  # Stack arrays along a new axis

    def _pad_arrays(self):
        """
        Pads each array to the full length by adding zeros at the end.
        """
        self.array = np.pad(self.array, ((0, 0), (0, 0), (0, self.spacing)), mode='constant', constant_values=0)

    def reset(self):
        if self._starting_class is None:
            self._current_class = self.rng.integers(0, self.num_classes)
        else:
            if self._starting_class < 0 or self._starting_class >= self.num_classes:
                raise ValueError(f"Starting class must be between 0 and {self.num_classes - 1}")
            self._current_class = self._starting_class
        self.count = 0
        self._finished = False

    def switch(self):
        r = self.rng.random()
        if r < self.p:
            self._current_class = (self._current_class + 1) % self.num_classes

    def generate(self) -> np.ndarray:
        idx = self.count % self._full_length

        # When spiking pattern is finished
        if self.count >= (self._pattern_length - 1):
            self._finished = True

        # When waiting time is finished, reset to initial state
        if self.count >= self._full_length:
            self._finished = False
            self.count = 0
            self.switch()

        # Increment count after resetting state
        self.count += 1

        # Slice stored array using recently resetted class
        spikes = self.array[self._current_class, :, idx]
        return spikes
    
    # def get_label(self) -> int:
    #     """
    #     Returns the current class label.
    #     """
    #     return self.current_class

    # @property
    # def finished(self) -> bool:
    #     """
    #     Returns whether the current pattern is finished.
    #     """
    #     return self._finished


class CustomTimingGenerator(SpikeGenerator):
    """
    A spike generator that takes in a list of classes (represented as timings) and perform modulations (jittering, failure-rate) during generation of each sample.
    """
    patterns: List[np.ndarray]
    
    def __init__(self, input_size: int, duration: int, patterns: List[np.ndarray] = None, labels: List[int] = None, *, spacing: int = None,
                 failure_rate: float = 0.0, jitter_std: int = 0, randomise_class: bool = True,
                 timings: List[np.ndarray] = None,
                 seed=None):
        super().__init__(input_size, seed)
        # Base parameters
        self.spacing = max(0, int(spacing)) if spacing is not None else 0
        self._pattern_length = duration
        self._full_length = self._pattern_length + self.spacing

        # Init classes separately if specified, otherwise use length of timing
        if labels is not None:
            self._validate_labels(patterns, labels)
        self.num_classes = len(patterns) if labels is None else len(set(labels))
        self.labels = labels if labels is not None else list(range(self.num_classes))
        self.num_samples = len(patterns)

        # Make a copy of raw timing data to prevent modifications like jittering
        if timings is not None:
            patterns = timings # Backward compatibility
        self.patterns = patterns.copy()
        self._validate_timings()
        self.array = np.zeros((input_size, self._full_length), dtype=np.int8)

        # Perturbation parameters
        self.failure_rate = min(1.0, max(0.0, failure_rate))
        self._failure = True if self.failure_rate > 0 else False
        self.jitter_std = np.abs(jitter_std)
        self._jitter = True if self.jitter_std > 0 else False
        self._static = True if (self._failure and self._jitter) else False
        self.randomise_class = randomise_class

        # Tracking parameters
        self.count = 0
        self._finished = False
        self._current_class = None
        self._sample_id = None
        self.reset()

    def _validate_labels(self, timings, labels):
        assert len(labels) == len(timings), f"If specified, the labels array must have the same length as the timings array. Got {len(labels)} classes and {len(timings)} timings."
        assert all(label >= 0 for label in labels), "All labels must be non-negative integers."
        if isinstance(labels, np.ndarray):
            assert np.isdtype(labels.dtype, np.int_), "Labels must be integers."
        else:
            assert all(isinstance(label, int) for label in labels), "All labels must be integers."

    def _validate_timings(self):
        for timing in self.patterns:
            if timing.size == 0:
                # Ignore empty timings
                continue
            # First check for appropriate dimension
            assert np.ndim(timing) == 2, "Timings must be a list of 2D arrays."
            # Second check if second axis has length 2
            assert timing.shape[1] == 2, "Each timing must have two columns: (neuron_id, time_step)."
            # Third check if neuron_id is within range
            assert np.all(timing[:, 0] < self.input_size) and np.all(timing[:, 0] >= 0), f"Neuron IDs must be in range [0, {self.input_size - 1}]."
            # Fourth check if time_step is within range
            assert np.all(timing[:, 1] < self._pattern_length) and np.all(timing[:, 1] >= 0), f"Time steps must be in range [0, {self._pattern_length - 1}]."
            # Fifth check if pairings are unique
            assert np.all(np.unique(timing, axis=0, return_counts=True)[1] == 1), "Each (neuron, time) pairing must be unique."

    def update_classes(self, timings: List[np.ndarray], labels: List[int] = None) -> None:
        """
        Updates the timings with a new list of timings.
        """
        if labels is not None:
            self._validate_labels(timings, labels)
            self.labels = labels

        self.patterns = timings.copy()
        self._validate_timings()

    def reset(self):
        self.count = 0
        self._finished = False
        self.array.fill(0)
        self._current_class = 0
        self._sample_id = 0
        # pattern_id = self.rng.integers(0, self.num_samples) if self.randomise_class else 0
        pattern_id = self.sample_pattern_id(reset=True)
        self.setup_array(pattern_id)

    def setup_array(self, pattern_id: int):
        """
        Intrenally fill up array based on timings of chosen class.
        """
        self.array.fill(0)

        # Sample pattern and labels by sample id

        class_id = self.labels[pattern_id]
        pattern = self.patterns[pattern_id].copy()

        # Record id's
        self._current_class = class_id
        self._sample_id = pattern_id

        # Check if timings is empty
        if pattern.size == 0:
            return

        # Apply spike failure
        if self._failure:
            len_timings = pattern.shape[0]
            pattern = pattern[self.rng.binomial(1, p=1 - self.failure_rate, size=len_timings).astype(bool)]
        # Apply jitter
        if self._jitter:
            len_timings = pattern.shape[0] # Length may be reduced after applying failure
            # Sample timiing deviation from normal distribution
            jitter = np.round(self.rng.normal(0, scale=self.jitter_std, size=len_timings)).astype(np.int_)
            # Trim to make sure timings is within range
            pattern[:, 1] = np.clip(pattern[:, 1] + jitter, 0, self._pattern_length - 1)

        # Fill up array with spikes
        np.put(self.array, np.ravel_multi_index(pattern.T, self.array.shape), 1)

    def sample_pattern_id(self, reset: bool = False) -> int:
        if self.randomise_class:
            pattern_id = self.rng.integers(0, self.num_samples)
        else:
            if reset:
                pattern_id = 0
            else:
                pattern_id = (self._sample_id + 1) % self.num_samples
        return pattern_id

    def switch(self):
        pattern_id = self.sample_pattern_id()
        self.setup_array(pattern_id)

    def generate(self) -> np.ndarray:
        if self._current_class is None:
            self.reset()

        # Check if the current pattern is finished and switch classes if necessary
        if self.count >= self._full_length:
            self._finished = False
            self.count = 0
            self.switch()

        # Generate spikes for the current time step
        idx = self.count % self._full_length
        spikes = self.array[:, idx]#.copy()  # Avoid using copy to preserve memory

        # Increment count after generating spikes
        self.count += 1

        # Mark as finished if the current pattern is completed
        if self.count >= self._pattern_length:
            self._finished = True

        return spikes
        

def construct_linear_pattern_timing(input_size, interval, ascending: bool = True) -> np.ndarray:
    """
    Returns timing pairings (neuron_id, time_step) for a pattern of spikes representing a monotonic linear pattern, staggered by `interval` time steps.
    """
    if ascending:
        return np.array([
            (n, n * interval) for n in range(input_size)
        ])
    else:
        return np.array([
            (input_size - n - 1, n * interval) for n in range(input_size)
        ])

def create_binary_class_timing(input_size, interval) -> List[np.ndarray]:
    """
    Returns a list of two timing arrays for binary classification tasks.
    The first array represents an ascending pattern and the second represents a descending pattern.
    Each array contains timing pairs (neuron_id, time_step) for spikes.
    """
    return [
        construct_linear_pattern_timing(input_size, interval, ascending=True),
        construct_linear_pattern_timing(input_size, interval, ascending=False)
    ]

def create_binary_class_array(input_size, interval, failure_rate=0.0, jitter=0) -> List[np.ndarray]:
    """
    Returns a list of two arrays for binary classification tasks.
    Each array contains spikes in the form of a 2D numpy array with shape (input_size, pattern_length).
    The first array represents an ascending pattern and the second represents a descending pattern.
    """
    A = construct_array(input_size, interval, ascending=True, failure_rate=failure_rate, jitter=jitter)
    B = construct_array(input_size, interval, ascending=False, failure_rate=failure_rate, jitter=jitter)
    return [A, B]


# Factory function to create a spike generator based on class name and input size.
def create_spikegen(class_name, input_size, binary:bool=True, **kwargs):
    """
    Creates an instance of a spike generator based on the specified class name.

    Args:
        class_name (str): The name of the spike generator class to instantiate.
        input_size (int): The size of the input for the spike generator.
        **kwargs: Additional keyword arguments to pass to the spike generator class.

    Returns:
        SpikeGenerator: An instance of the specified spike generator class.

    Raises:
        ValueError: If the specified class name is not found or does not inherit from SpikeGenerator.

    Notes:
        - If `class_name` is "CustomTimingGenerator", the function generates binary class timings
          and calculates the duration based on the `interval` argument.
        - If `class_name` is "CustomArrayGenerator", the function generates binary class arrays
          based on the `interval` argument.
        - For other class names, the function directly instantiates the class if it is a subclass
          of `SpikeGenerator`.
    """
    spikegen_cls = globals().get(class_name, None)
    binary = kwargs.pop("binary", binary)
    if class_name == "CustomTimingGenerator" and binary:
        interval = kwargs.pop("interval", 1)
        timings = create_binary_class_timing(input_size, interval)
        duration = (input_size - 1) * interval + 1
        spikegen = spikegen_cls(input_size, duration, timings, **kwargs)
    elif class_name == "CustomTimingGenerator" and not binary:
        spikegen = spikegen_cls(input_size, **kwargs)
    elif class_name == "CustomArrayGenerator" and binary:
        interval = kwargs.pop("interval", 1)
        arrays = create_binary_class_array(input_size, interval)
        spikegen = spikegen_cls(input_size, arrays=arrays, **kwargs)
    elif class_name is not None and issubclass(spikegen_cls, SpikeGenerator):
        spikegen = spikegen_cls(input_size, **kwargs)
    else:
        raise ValueError(f"Spike generator class {class_name} not found or does not inherit from SpikeGenerator.")
    return spikegen


def construct_poisson_spike_times_1(r, dt, T, rng, n):
    """
    Threshold-based Poisson spike time generation. Random numbers for determining spike sampled from [0, 1] Uniform distribution.

    1. Sample random numbers uniformly in [0, 1] in shape (n, T)
    2. Determine if spike occurs according to threshold of r * dt
    3. Determine timestep where spikes occur in each neuron, and return (neuron, time) pairs
    """
    x = rng.random((n, T))
    sp = (x < r * dt).astype(int)
    # times = [(nid, t) for nid, t in zip(*np.nonzero(sp))]
    times = np.stack(np.nonzero(sp), axis=1)
    return times

def construct_poisson_spike_times_2(r, dt, T, rng, n):
    """
    Interval-based Poisson spike time generation. Intervals between spikes sampled from exponential distribution.

    1. Sample spike interval from exponential
    2. Cumulatively sum intervals to get spike times
    3. Cut off spike times beyond T
    4. Concatenate into (neuron, time) pairs
    """
    x = rng.exponential(1 / (r*dt), (n, T))
    xc = np.cumsum(x, axis=1).round().astype(int)
    times = np.array([(i, t) for i, tm in enumerate(xc) for t in np.unique(tm) if t < T], dtype=np.int32)
    return times

def construct_poisson_spike_times_3(r, dt, T, rng, n):
    """
    Count-based Poisson spike time generation. Counts for each neuron sampled from Poisson distribution.
    
    1. Sample number of spikes for each neuron from Poisson distribution
    2. Sample spike times uniformly in [0, T] according to the number of spikes for each neuron
    3. Return spike times as (neuron, time) pairs
    """
    counts = rng.poisson(T * dt * r, n)
    times = [np.sort(np.unique(rng.uniform(0.0 + dt, T-0.5, size=c).round())) for c in counts]
    # times = [tm - tm % dt for tm in times]
    times = np.array([(i, t) for i, tm in enumerate(times) for t in tm], dtype=np.int32)
    return times

_poisson_dict = {
    "threshold": construct_poisson_spike_times_1,
    "interval": construct_poisson_spike_times_2,
    "count": construct_poisson_spike_times_3
}

def create_poisson_class_timing(input_size, duration, rate, *, 
                                dt=1e-3, num_classes=2, num_sets: int= 1, simplify: bool= True,
                                rng: np.random.Generator=None,
                                method: Literal["threshold", "interval", "count"] = "threshold") -> List[np.ndarray]:
    """
    Creates a list of Poisson spike timings for classification tasks.
    Each class has a different rate and the timings are generated based on the specified duration and dt.
    
    Args:
        input_size (int): The number of neurons in the input layer.
        duration (int): The total duration of the spike pattern.
        rate (float): The average firing rate of spikes per neuron.
        dt (float): The time step size.
        num_classes (int): The number of classes of patterns within a set.
        num_sets (int): The number of sets of num_classes patterns to generate.
        simplify (bool): If True (default), for num_sets=1, returns a single list of length num_classes.

    Returns:
        List[np.ndarray]: A list of timing arrays for each class, where each array contains (neuron_id, time_step) pairs.
    """
    if rng is None:
        rng = np.random.default_rng()
    func = _poisson_dict.get(method, construct_poisson_spike_times_1)
    sets = []
    for s in range(num_sets):
        # Generate timings for each class
        in_sets = []
        for i in range(num_classes):
            times = func(rate, dt, duration, rng, input_size)
            in_sets.append(times)
        sets.append(in_sets)
    if simplify and num_sets == 1:
        return sets[0]
    else:
        return sets

def create_poisson_patterns_and_labels(
    input_size: int, duration: int, rate: float, *,
    dt: float = 1e-3, num_classes: int = 2, num_stimuli: int = 5,
    rng: np.random.Generator = None,
    method: Literal["threshold", "interval", "count"] = "threshold"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Creates a list of $$M$$ Poisson spike patterns and $$C$$ randomised labels.
    
    Args:
        input_size (int): The number of neurons in the input layer.
        duration (int): The total duration of the spike pattern.
        rate (float): The average firing rate of spikes per neuron.
        dt (float): The time step size.
        num_classes (int): $$C$$, The number of class labels to assign to each stimuli.
        num_stimuli (int): $$M$$, The total number of patterns to generate (regardless of number of class labels).
        rng (np.random.default_rng): Random number generator instance.
        method (str): Method to use for generating Poisson spike timings.

    Returns:
        patterns, labels (List[np.ndarray], np.ndarray):
            - patterns: A list of spike patterns, where each pattern is a 2D array of shape (input_size, duration).
            - labels: A numpy array of randomised class labels for each pattern.
    """
    if rng is None:
        rng = np.random.default_rng()
    patterns = create_poisson_class_timing(input_size, duration, rate, dt=dt, 
                                           num_classes=num_stimuli, num_sets=1,
                                           simplify=True, rng=rng, method=method)

    # Generate random labels for each timing
    labels = rng.binomial(num_classes - 1, p=0.5, size=num_stimuli)

    return patterns, labels


if __name__ == "__main__":
    gen = PatternSpikeGenerator(input_size=5, interval=2, spacing=5, ascending=False, start_spike=True)
    N = len(gen)
    result = []
    gen.reset()
    for i in range(N*2):
        spk = gen.generate()
        print(spk)
        print(f"t: {i:2}, Finished: {gen.finished}, Delay Count: {gen.delay_count}, Spike Count: {gen.spike_count}", end='\n\n')