import numpy as np
from typing import List, Literal
from .base import NeuronLayerProtocol


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

def trace_x3(t, dt, tau, A):
    "Goes up to a fixed value 1"
    t = t * dt
    x = A * np.exp(-t / tau)
    return x

def softmax(x, temperature=1.0):
    ex = np.exp(x / temperature)
    return ex / np.sum(ex)

class NeuronLayer(NeuronLayerProtocol):
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
    def __init__(self, size: int, *, tau_mem: float = None, tau_trace: float = None, dt: float = 1e-3, threshold: float = 1.0, 
                 wta: bool = False, sim_method: Literal["event-driven", "step-wise"] = "step-wise",
                 spike_method: Literal["deterministic", "stochastic"] = "deterministic", 
                 softmax_temp: float = 1.0, 
                 spike_condition: Literal["every", "input"] = "every",
                 reset_mechanism: Literal["rest", "zero", "subtract"] = "rest", mem_rest: float = 0.0, 
                 reset_condition: Literal["only-winner", "all-above", "all", "none"] = "all-above", delayed_wta: bool = False, 
                 trace_amp: float = 1.0, #trace_type: Literal["dx1", "dx2", "dx3", "dx4"] = None,
                 trace_type: Literal["cumulative", "recent", "dx1", "dx2", "dx3", "dx4"] = "recent"):
        # Simulation parameters
        self.size = size
        self.dt = dt
        self.sim_method = sim_method
        self._event_driven = sim_method == "event-driven"
        self._step_wise = sim_method == "step-wise"
        self._last_only = trace_type == "recent" 
        if trace_type.startswith("dx"): # Backwards compatibility
            self._last_only = True if trace_type in ["dx3", "dx4"] else False # dx2 and dx1 are cumulative traces

        # Deals with positive integer tau's as a unit of dt
        if tau_mem is not None and isinstance(tau_mem, int):
            tau_mem = tau_mem * dt
        if tau_trace is not None and isinstance(tau_trace, int):
            tau_trace = tau_trace * dt

        # Membrane potential parameters
        self.mem_rest = mem_rest
        self.tau_mem = tau_mem if tau_mem is not None else tau_trace if tau_trace is not None else dt
        self.beta_mem = np.exp(-self.dt / self.tau_mem) # Decay rate
        # Trace parameters
        self.tau_trace = tau_trace if tau_trace is not None else tau_mem if tau_mem is not None else dt
        self.trace_amp = trace_amp
        self.beta_trace = np.exp(-self.dt / self.tau_trace)  # Decay rate for trace

        # Spiking parameters
        self._spike_method = spike_method
        self._stochastic_spike = spike_method == "stochastic"
        self._initial_softmax_temp = softmax_temp
        self._softmax_temp = softmax_temp
        self.wta = wta if not self._stochastic_spike else True # Assume winner take all when spiking is stochastic
        # Spike condition
        self.spike_condition = spike_condition
        self._spike_cond_every = spike_condition == "every"
        self._spike_cond_input = spike_condition == "input"
        
        # Reset parameters
        self.threshold = threshold
        # Reset condition
        if delayed_wta == True:
            reset_condition = "only-winner" # Backwards compatibility
        self.reset_condition = reset_condition
        self._reset_cond_one = reset_condition == "only-winner"
        self._reset_cond_abv = reset_condition == "all-above"
        self._reset_cond_all = reset_condition == "all"
        self._reset_cond_nan = reset_condition == "none"
        # Reset mechanism
        self.reset_mechanism = reset_mechanism
        if self.reset_mechanism == "zero":
            self.reset_mechanism = "rest"
            self.mem_rest = 0.0
        self._reset_mech_rest = self.reset_mechanism == "rest"
        self._reset_mech_subt = self.reset_mechanism == "subtract"

        # Membrane potential
        self.membrane = np.full((size,), mem_rest)
        # Spike status
        self.spike = np.zeros(size, dtype=np.int8)
        if self._event_driven:
            # Time since last spike
            self.tssp = np.full(size, dtype=np.float32, fill_value=np.inf)
            self.last_peak = np.zeros(size, dtype=np.float32)
        elif self._step_wise:
            # Trace
            self._trace = np.zeros(size, dtype=np.float32)

    def reset(self):
        """
        Reset the neuron layer state.
        """
        self.membrane.fill(self.mem_rest)
        self.spike.fill(0)
        if self._event_driven:
            self.tssp.fill(np.inf)
            self.last_peak.fill(0.0)
        elif self._step_wise:
            self._trace.fill(0.0)

    def forward(self, input_current: np.ndarray):
        """
        Update the neuron layer state based on the input current and time step.
        """
        # Reset the membrane potential for spiking neurons
        self._reset_membrane(input_current)
        # Calculate the new membrane potential
        self._update_membrane(input_current)
        # Check for spikes
        self._set_spike(input_current)
        # Update the time since last spike
        # self._update_tssp()
        # Update trace
        self._update_trace()

        return self.spike.astype(np.int8)

    def _reset_membrane(self, input_current: np.ndarray = None):
        # If reset_condition == "only-winner", 
        #   only the neuron that spiked (winner) gets their membrane reset.
        #   Other non-spiking neurons with membranes above threshold, will automatically spike in the next step (hence, delayed)
        # If reset_condition == "all-above", 
        #   all neurons with membranes above thresholds get reset, regardless of whether or not they spiked.
        # If reset_condition == "all",
        #   all neurons get reset, regardless of their previous membrane potentials.
        cond = self.spike.astype(bool) if self._reset_cond_one else \
                (self.membrane >= self.threshold) if self._reset_cond_abv else \
                sum(self.spike) > 0 if self._reset_cond_all else \
                False if self._reset_cond_nan else \
                False
        if self._reset_mech_rest:
            self.membrane[cond] = self.mem_rest
        elif self._reset_mech_subt:
            self.membrane[cond] -= self.threshold

    def _update_membrane(self, input_current):
        self.membrane = self.beta_mem * (self.membrane - self.mem_rest) + input_current + self.mem_rest

    def _set_spike(self, input_current: np.ndarray = None):
        if self._spike_cond_input:
            if input_current is not None and sum(input_current) == 0:
                self.spike.fill(0)
                return
        # Deterministic spiking
        if not self._stochastic_spike:
            if self.wta:
                above_thr = self.membrane >= self.threshold
                self.spike.fill(0)
                if sum(above_thr) == 0:
                    return
                else:
                    idx = np.argmax(self.membrane)
                    self.spike[idx] = 1
                    return
            else:
                self.spike = (self.membrane >= self.threshold).astype(np.int8)
        # Stochastic spiking
        else:
            # Assumes WTA by default -> only one choice of neuron can spike stochastically
            above_thr = self.membrane >= self.threshold
            self.spike.fill(0)
            if sum(above_thr) == 0:
                # Will only spike if at least one neuron has membrane above threshold
                return
            else:
                # Probability of spiking is based on membrane potentials of neurons themselves (softmax with temperature),
                #   regardless of which one is above threshold
                probs = softmax(self.membrane, temperature=self._softmax_temp)
                idx = np.random.choice(self.size, p=probs)
                self.spike[idx] = 1


    def _update_tssp(self):
        # if self._event_driven:
        # Get indices of only neurons that spike, instead of performing calc on whole array
        spk_idx = self.spike.nonzero()[0]
        # For those that spikes, update peak _before_ updating tssp (since we want to get decayed value before rise)
        if self._last_only:
            # Most recent
            self.last_peak[spk_idx] = self.trace_amp
        else:
            # Cumulative trace
            self.last_peak[spk_idx] = trace_x3(self.tssp[spk_idx], self.dt, self.tau_trace, self.last_peak[spk_idx]) + self.trace_amp
        # Update time since last spike
        self.tssp += 1
        self.tssp[spk_idx] = 0

        # elif self._step_wise:
        #     if self.trace_type == "dx3":
        #         self.tssp = np.where(self.spike, 0, self.tssp + 1)

    def _update_trace(self):
        """
        Update the trace based on the time since last spike and the trace type.
        """
        if self._event_driven:
            self._update_tssp()
        elif self._step_wise:
            if self._last_only:
                # Most recent
                self._trace = np.where(self.spike, self.trace_amp, self._trace * self.beta_trace)
            else:
                # Cumulative
                decay = self._trace * self.beta_trace
                self._trace = np.where(self.spike, self.trace_amp + decay, decay)
            # if self.trace_type == "dx1":
            #     self._trace = self._trace + trace_dx1(self._trace, self.tau_trace/self.dt, self.spike, self.trace_amp)
            # elif self.trace_type == "dx2":
            #     self._trace = self._trace + trace_dx2(self._trace, self.tau_trace/self.dt, self.spike, self.trace_amp)
            # elif self.trace_type == "dx3":
            #     self._trace = trace_x3(self.tssp, self.dt, self.tau_trace, self.trace_amp)
            # elif self.trace_type == "dx4":
            #     self._trace = np.where(self.spike, self.trace_amp, self._trace * self.beta_trace)
        
    # def get_trace(self):
    #     """
    #     Calculate the trace of the neuron layer based on the time since last spike.
    #     """
    #     t = self.dt * self.tssp
    #     return self.trace_amp * np.exp(-t / self.tau_trace)

    def get_trace(self, idx: List[int] | np.ndarray[int] = None) -> np.ndarray:
        if self._step_wise:
            return self._trace
        elif self._event_driven:
            if idx is None:
                return trace_x3(self.tssp, self.dt, self.tau_trace, self.last_peak)
            else:
                return trace_x3(self.tssp[idx], self.dt, self.tau_trace, self.last_peak[idx])

    @property
    def trace(self) -> np.ndarray:
        """
        Returns the trace of the neuron layer.
        """
        return self.get_trace()

    @property
    def spike_method(self) -> str:
        return self._spike_method
    
    @spike_method.setter
    def spike_method(self, value: Literal["stochastic", "deterministic"]):
        assert value in ["stochastic", "deterministic"], "New spike method model not supported."
        self._spike_method = value
        self._stochastic_spike = self._spike_method == "stochastic"

    @property
    def softmax_temp(self) -> float:
        if self._stochastic_spike:
            return self._softmax_temp
        else:
            return 0.0
    
    @softmax_temp.setter
    def softmax_temp(self, value: float):
        self._softmax_temp = value

    def __repr__(self):
        return f"NeuronLayer(size={self.size}, tau_mem={self.tau_mem}, tau_trace={self.tau_trace}, threshold={self.threshold}, wta={self.wta})"
    def __str__(self):
        return self.__repr__()
    