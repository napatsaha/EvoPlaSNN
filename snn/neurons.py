import numpy as np
from typing import List, Literal


def leaky_integrate_and_fire(membrane, input_current, beta, threshold):
    spk = (membrane > threshold).astype(np.int8)

    membrane = beta * membrane + input_current - spk * threshold

    return membrane, spk

def trace_dx1(x, tau, spk, A=1.0):
    "Goes up by fixed magnitude A"
    dx = - x / tau + A * spk
    return dx

def trace_dx2(x, tau, spk, A):
    "Saturates between 0 and 1"
    dx = - x / tau + spk * A * (1 - x)
    return dx

def trace_dx3(x, tau, spk):
    "Goes up to a fixed value 1"
    dx = trace_dx2(x, tau, spk, A=1.0)
    return dx


class NeuronLayer:
    """
    A class represnting a single vertical layer of spiking neurons.
    Must store:
    - membrane potential
    - whether neuron is spiking or not
    - time since last spike
    each of these as an array the size of N (N = number of neurons in the layer)

    The reason this is a layer instead of single neurons is so that operations can be performed on entire arrays,
    assuming a regular step-wise approach to simulating SNN.

    Can also get the trace which will convert the time since last spike to a value that decays since the last spike.
    Assumming each new spike resets the trace to its maximum value.
    """
    def __init__(self, size: int, *, tau_mem: float = 1e-1, tau_trace: float = 1e-1, dt: float = 1e-1, threshold: float = 1.0, 
                 membrane_start: float = 0.0, reset_mechanism: Literal["zero", "subtract"] = "zero",
                 trace_amp: float = 1.0, trace_type: Literal["dx1", "dx2", "dx3"] = "dx3"):
        # Basic parameters
        self.size = size
        self.dt = dt

        # Membrane potential parameters
        self.membrane_start = membrane_start
        self.tau_mem = tau_mem
        self.beta_mem = np.exp(-dt / tau_mem) # Decay rate
        # Trace parameters
        self.tau_trace = tau_trace
        self.trace_amp = trace_amp
        self.trace_type = trace_type
        # Threshold parameters
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism if reset_mechanism in ["zero", "subtract"] else "zero"

        # Membrane potential
        self.membrane = np.full((size,), membrane_start)
        # Spike status
        self.spike = np.zeros(size, dtype=np.int8)
        # Time since last spike
        self.tssp = np.zeros(size, dtype=np.int32)
        # Trace
        self.trace = np.zeros(size, dtype=np.float32)

    def forward(self, input_current: np.ndarray):
        """
        Update the neuron layer state based on the input current and time step.
        """
        # Reset the membrane potential for spiking neurons
        if self.reset_mechanism == "zero":
            self.membrane = np.where(self.spike, 0, self.membrane)
        elif self.reset_mechanism == "subtract":
            self.membrane = np.where(self.spike, self.membrane - self.threshold, self.membrane)
        # Calculate the new membrane potential
        self.membrane = self.beta_mem * self.membrane + input_current
        # Check for spikes
        self.spike = (self.membrane >= self.threshold)
        # Update the time since last spike
        self.tssp = np.where(self.spike, 0, self.tssp + 1)

        return self.spike.astype(np.int8)

    def update_trace(self):
        """
        Update the trace based on the time since last spike and the trace type.
        """
        if self.trace_type == "dx1":
            self.trace += trace_dx1(self.trace, self.tau_trace/self.dt, self.spike, self.trace_amp)
        elif self.trace_type == "dx2":
            self.trace += trace_dx2(self.trace, self.tau_trace/self.dt, self.spike, self.trace_amp)
        elif self.trace_type == "dx3":
            self.trace += trace_dx3(self.trace, self.tau_trace/self.dt, self.spike)

        return self.trace

    # def get_trace(self):
    #     """
    #     Calculate the trace of the neuron layer based on the time since last spike.
    #     """
    #     t = self.dt * self.tssp
    #     return self.trace_amp * np.exp(-t / self.tau_trace)

    def get_trace(self):
        return self.trace

    def __repr__(self):
        return f"NeuronLayer(size={self.size}, tau_mem={self.tau_mem}, tau_trace={self.tau_trace}, dt={self.dt}, threshold={self.threshold})"
    def __str__(self):
        return self.__repr__()
    