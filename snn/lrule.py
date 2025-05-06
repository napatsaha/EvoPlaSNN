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
        pass


class Empty_Rule(LearningRule):
    """
    A dummy learning rule that does nothing.
    """
    def __init__(self):
        super().__init__()

    def update(self, w: np.ndarray, **kwargs) -> np.ndarray:
        # No update
        return w


class STDP_Rule(LearningRule):
    def __init__(self, mu, lambd, alpha, dt):
        super().__init__()
        self.mu = mu
        self.lambd = lambd
        self.alpha = alpha
        # self.tau_trace = tau_trace
        self.dt = dt

    def F_neg(self, w):
        return self.lambd * self.alpha * (w**self.mu)
    
    def F_pos(self, w):
        return self.lambd * (1 - w)**self.mu

    def stdp_update(self, w, spk_pre, spk_post, trace_pre, trace_post):
        """
        Update the weights based on the STDP rule.
        """
        # Compute the weight change
        delta_w = self.F_pos(w) * spk_post * trace_pre - self.F_neg(w) * spk_pre * trace_post
        return delta_w
        
    def update(self, w, spk_pre, spk_post, trace_pre, trace_post) -> np.ndarray:
        dw = self.stdp_update(w, spk_pre, spk_post, trace_pre, trace_post)
        # Update the weights
        w = np.clip(w + dw, 0, 1)
        return w


