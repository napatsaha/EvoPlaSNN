from typing import Literal
import numpy as np
from abc import ABC, abstractmethod

from .base import LearningRule


class STDP_Rule(LearningRule):
    def __init__(self, mu, lambd, alpha, dt, *,
                #  w_min: float = 0.0, w_max: float = 1.0, clip_w: bool = True,
                  condition: Literal["on-spike", "on-reward"] = "on-spike"):
        super().__init__()
        self.mu = mu
        self.lambd = lambd
        self.alpha = alpha
        # self.tau_trace = tau_trace
        self.dt = dt
        # self.w_min = w_min
        # self.w_max = w_max
        # self.clip_w = clip_w
        assert condition in ("on-spike", "on-reward"), "Condition must be either 'on-spike' or 'on-reward'"
        self.condition = condition

    def F_neg(self, w):
        return self.lambd * self.alpha * (w**self.mu)
    
    def F_pos(self, w):
        return self.lambd * (1 - w)**self.mu

    def compute_dw(self, w, spk_pre, spk_post, trace_pre, trace_post):
        """
        Update the weights based on the STDP rule.
        """
        # Compute the weight change
        if spk_post.any():
            # LTP = self.F_pos(w) * spk_post * trace_pre
            LTP = np.where(spk_post, self.F_pos(w) * trace_pre, 0)
        else:
            LTP = 0
        if spk_pre.any():
            # LTD = self.F_neg(w) * spk_pre * trace_post
            LTD = np.where(spk_pre, self.F_neg(w) * trace_post, 0)
        else:
            LTD = 0
        delta_w = LTP - LTD

        # delta_w = self.F_pos(w) * spk_post * trace_pre - self.F_neg(w) * spk_pre * trace_post
        return delta_w
        
    def update(self, w, spk_pre, spk_post, trace_pre, trace_post, reward=None) -> np.ndarray:
        # Update the weights
        if self.condition == "on-spike":
            dw = self.compute_dw(w, spk_pre, spk_post, trace_pre, trace_post)
        elif self.condition == "on-reward":
            if bool(reward):
                dw = self.compute_dw(w, spk_pre, spk_post, trace_pre, trace_post)
            else:
                dw = 0

        w = w + dw        

        # Clip the weights
        # if self.clip_w:
        #     w = np.clip(w, self.w_min, self.w_max)
        return w