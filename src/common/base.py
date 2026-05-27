from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

from typing import Any, Literal, Protocol, List, Sequence, Tuple, Dict, Union


class NeuronLayerProtocol(Protocol):
    """
    Protocol class for NeuronLayer.
    Defines the public attributes and methods with their descriptions.
    """

    size: int
    dt: float
    membrane: np.ndarray
    spike: np.ndarray
    tssp: np.ndarray
    _trace: np.ndarray

    def __init__(self, size: int, *, tau_mem: float, tau_trace: float, dt: float, threshold: float,
                 wta: bool, membrane_start: float, reset_mechanism: Literal["zero", "subtract"],
                 trace_amp: float, trace_type: Literal["dx1", "dx2", "dx3"]) -> None:
        """
        Initialize the neuron layer with the given parameters.
        """
        pass

    def reset(self) -> None:
        """
        Reset the neuron layer state, including membrane potential, spike status, time since last spike, and trace.
        """
        pass

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """
        Update the neuron layer state based on the input current and time step.
        Returns the spike status as an array.
        """
        pass

    def get_trace(self) -> np.ndarray:
        """
        Return the trace of the neuron layer, which decays since the last spike.
        """
        pass

    def update_thresholds(self, delta_thr: np.ndarray):
        """
        Update the firing thresholds of the neurons by adding delta_thr.
        """
        pass


# from .snn import NeuronLayer


class LearningRule(ABC):
    """
    Abstract base class for learning rules.
    """
    parameters: np.ndarray | List
    input_size: int
    output_size: int
    input_order: Sequence
    
    def __init__(self):
        pass

    @abstractmethod
    def update(self, synapse: 'SynapseLayerProtocol', reward: float, always_return_tuple: bool) -> Union[np.ndarray, Tuple]:
        """
        Update the synaptic weights based on the learning rule, given a Synapse layer and reward

        Args:
            synapse (SynapseLayerProtocol): Synapse Layer which contains all necessary information for the LearningRule
                to extract inputs from
            reward (float): The only global input needed for the LearningRule
            always_return_tuple (bool): If True, output will be Tuple of (delta_weight, delta_threshold)

        Returns:
            Union[np.ndarray, Tuple]: If `always_return_tuple=False` will only return either delta_weight or delta_threshold.
                Otherwise will return both as a tuple
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """
        Calculate outputs of a rule's inner function (not necessarily the same as update)

        Args:
            inp (np.ndarray): Inputs (Must be compatible with Learning Rule's input_size)

        Returns:
            np.ndarray: Output array
        """

class SynapseLayerProtocol(Protocol):
    """
    Protocol abstract class for SynapseLayer.
    Defines the method and attribute names along with their descriptions.
    """

    pre_layer: NeuronLayerProtocol
    post_layer: NeuronLayerProtocol
    weights: np.ndarray
    learning_rule: LearningRule
    eligibility_trace: np.ndarray | None

    def __init__(self, pre_layer: NeuronLayerProtocol, post_layer: NeuronLayerProtocol, *,
                 learning_rule: LearningRule, eligibility_trace: bool, tau_syn: float, dt: float,
                 weight_init: str, weight_init_params: dict, weight_min: float, weight_max: float,
                 clip_weights: bool, normalise_weights: bool, normalise_method: Literal["sum", "L2", "P"],
                 normalise_params: dict) -> None:
        """
        Initialize the SynapseLayer with the given parameters.
        """
        pass

    def forward(self, spike_input: np.ndarray) -> np.ndarray:
        """
        Compute the output current to the next neuron layer given a spike current input from the previous neuron layer.
        """
        pass

    def update(self, reward: float | None) -> None:
        """
        Update the synaptic weights based on the learning rule.
        """
        pass

    def reset(self) -> None:
        """
        Reset the synaptic weights to their initial state.
        """
        pass

    def update_eligibility_trace(self) -> None:
        """
        Update the eligibility trace based on the pre and post neuron layer spikes.
        """
        pass


class SNNProtocol(Protocol):
    """
    Spiking Neural Network Protocol
    """


class SimulatorProtocol(Protocol):
    """
    SNN Simulator Protocol

    Contains SNN, environment, etc
    """

    network: Any # Will fix later
    environment: Any # Will fix later


class Solver(Protocol):
    """
    Base solver for all Evolutionary Algorithms.
    """
    def __init__(self, popsize: int):
        pass

    def ask(self) -> List:
        """
        Returns and records (internally) a set of solutions.
        """
        raise NotImplementedError("ask method must be implemented by subclasses.")

    def tell(self, fitnesses: List):
        """
        Informs current solutions with evaluted fitnesses.
        """
        raise NotImplementedError("tell method must be implemented by subclasses.")

    def result(self) -> Tuple[object, float]:
        """
        Returns the best solutions and their fitnesses.
        """
        raise NotImplementedError("result method must be implemented by subclasses.")


class Evaluator(Protocol):
    """
    Base class for evaluation functions.
    """
    measure_behaviour: bool

    def __init__(self):
        pass
        # self.fitnesses: List[float] = []

    def evaluate(self, solution: object, num_trials: int = None) -> Tuple[List, float, float, Sequence]:
        """
        Evaluates a given solution and returns its fitness.

        Returns a Tuple containing: fts_list, avg_fts, std_fts, behv
        - List of fitnesses for each trial
        - Average fitness
        - Standard Deviation of fitnesses
        - Behaviour measure (if measure_behaviour is enabled)
        """
        raise NotImplementedError("evaluate method must be implemented by subclasses.")

    def get_parameter_size(self) -> int:
        """
        Returns the size of the parameter space.
        """
        raise NotImplementedError("get_parameter_size method must be implemented by subclasses.")

    def setup_logger(self, log_file: str = None):
        """
        Sets up a logger for the evaluator.
        """
        pass

    # def generate_new_classes(self) -> None:
    #     """
    #     Update set of classes used for spike generation.
    #     Meant to be called at beginning of each generation.
    #     """
    #     pass

    def setup_generation(self, gen_count: int, **kwargs):
        """
        Sets up at the beginning of each generation.  
        To be called outside the class (i.e. by `Manager`), before whole population is to be evaluated.
        """
        pass

    def setup_individual(self, inv_count: int, **kwargs):
        """
        Sets up at the beginning of each individual evaluation.  
        To be called inside the class (within `evaluate()`), before trial evaluation loop is begun.
        """
        pass

    def setup_trial(self, trial_count: int, **kwargs):
        """
        Sets up at the beginning of each trial evaluation.  
        To be called inside the class (within `evaluate()`), at the start of each iteration of the trial loop.
        """
        pass


class Genome(ABC):
    """
    Base class to allow for genetic-related operations in evolutionary Solver.
    """
    # def __init__(self, **kwargs):
    #     super().__init__()
        # self._parameters = parameters
    # genes: List['Parameter']

    @abstractmethod
    def mutate(self, rate: float, **kwargs) -> 'Genome':
        """
        Create a modified copy of itself

        Args:
            rate (float): mutation rate
        """

    @abstractmethod
    def crossover(self, other: 'Genome', rate: float) -> 'Genome' | List['Parameter']:
        """
        Crossover between two Genomes
        """

    @property
    def parameters(self) -> np.ndarray:
        """
        Returns a 1D genetic blueprint of the genome
        """
        # return self._parameters 

    @property
    def size(self) -> int:
        """
        Returns the number of parameters that exists in the genome
        """
        # return len(self._parameters)

    # def __repr__(self) -> str:
        # return f"Genome({self.parameters})"

    

class Parameter(ABC):
    """
    A Parameter (aka Gene) is a component of a Genome with specific value type, bounds and distribution.
    """
    length: int
    value: np.typing.ArrayLike

    @abstractmethod
    def mutate(self, flags: bool | List | ArrayLike) -> 'Parameter':
        """
        Mutate and create a modified Paramter with the same distribution and bounds

        Args:
            flags (bool | List | ArrayLike): an array of boolean flags to determine which location of gene
            to modify. (Must have same length as `Parameter.length`). If a scalar boolean value is passed through,
            the entire length of Paramter will be affected altogether.

        Returns:
            Parameter: new Paramter
        """
    
    @abstractmethod
    def crossover(self, other: 'Parameter', flags: bool | List | ArrayLike) -> 'Parameter':
        """
        Perform a point crossover with another Parameter given an array of flags where crossover should occur.

        Args:
            other (Parameter): The other Parameter
            flags (bool | List | ArrayLike): where to recombine the genes

        Returns:
            Parameter: New recombined Parameter
        """
        
    @abstractmethod
    def copy(self) -> 'Parameter':
        """
        _summary_
        """
    
    @abstractmethod
    def to_dict(self) -> 'Dict':
        """
        _summary_
        """


