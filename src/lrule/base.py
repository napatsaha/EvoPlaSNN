from typing import Literal
import numpy as np
from abc import ABC, abstractmethod
# from .snn import NeuronLayer


class LearningRule(ABC):
    """
    Abstract base class for learning rules.
    """
    def __init__(self):
        pass

    @abstractmethod
    def update(self) -> np.ndarray:
        """
        Update the synaptic weights based on the learning rule.
        """
        raise NotImplementedError()


class Empty_Rule(LearningRule):
    """
    A dummy learning rule that does nothing.
    """
    def __init__(self):
        super().__init__()

    def update(self, w: np.ndarray, **kwargs) -> np.ndarray:
        # No update
        return w
    
