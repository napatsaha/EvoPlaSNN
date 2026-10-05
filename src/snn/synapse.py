from typing import Dict, Literal
import warnings

from common.base import LearningRule, NeuronLayerProtocol, SynapseLayerProtocol
import numpy as np
# from .neurons import NeuronLayer
from lrule import Empty_Rule
from lrule.dual import DualLearningRule
from genome.genome import EvolvableLearningRule
# from .utils import tile
from .utils import Array_FIFO

class SynapseLayer(SynapseLayerProtocol):
    """
    Purpose: To represent a collection of synapse that connects one layer of neurons to another.
    (Assuming a Sequential Linear Layer network with all-to-all connections)

    Functionality:
    `forward()`: computes the output current to next neuron layer given a spike current input 
    from previous neuron layer -- i.e. weighted spike current
    `update()`: Update the entire synaptic weight efficacies based on a particular LearningRule
    `reset()`: Reset the synaptic weights to their initial state.
    `eligibility_trace`: If using eligibility traces, return the eligibility trace.
    `update_eligibility_trace()`: Update the eligibility trace based on the pre and post neuron layer spikes.
    """
    post_layer: NeuronLayerProtocol
    pre_layer: NeuronLayerProtocol
    weights: np.ndarray
    _learning_rule: LearningRule
    _internal_rule: LearningRule
    _external_rule: LearningRule
    
    def __init__(self, pre_layer: NeuronLayerProtocol, post_layer: NeuronLayerProtocol, *, 
                 learning_rule: LearningRule = None, plastic: bool = True,
                 pre_trace: bool = False, post_trace: bool = False,
                 eligibility_trace: bool = False, eligibility_pre: bool = False, eligibility_post: bool = False,
                 eligibility_stdp: bool = False, ltd_coef: float = 1.0,
                 eligibility_custom: bool = False,
                 e_max: float = None, e_min: float = None,
                 tau_syn: float = None, tau_pre: float = None, tau_post: float = None,
                 tau_weight: float = None,
                 dt: float = 1e-3,
                 synaptic_delay: int = 0, 
                 mask_method: Literal["whole", "by_pre", "by_post"] = "whole",
                 inhibition_prop: float | int = None, excitation_prop: float | int = None, connectivity: float = None, 
                 sim_method: Literal["event-driven", "step-wise"] = "step-wise",
                 weight_init: Literal["uniform", "normal", "constant"] = "uniform",
                 weight_clip_min: float = None, weight_clip_max: float = None,
                 w_min: float = 0.0, w_max: float = 1.0,
                 clip_weights: bool = True, normalise_weights: bool = False, 
                 normalise_method: Literal["sum", "L2", "P"] = "sum", #normalise_params: dict = None,
                 **kwargs):

        # Simulation parameters
        self.dt = dt
        self.sim_method = sim_method
        self._event_driven = sim_method == "event-driven"
        self._step_wise = sim_method == "step-wise"

        # Connecting neuron layers
        self.pre_layer = pre_layer
        self.post_layer = post_layer

        # Learning-related params
        self.plastic = plastic
        self._internal_rule = None
        self._external_rule = None
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()

        # Interactions with learning rule
        self._out_weights = False
        self._out_thresholds = False
        self._out_eligibility = False

        # Synaptic Delay
        self.synaptic_delay = synaptic_delay
        self._apply_delay = synaptic_delay > 0
        if self._apply_delay:
            self.current_buffer = Array_FIFO(shape=(self.post_layer.size, ), size=synaptic_delay + 1)

        # Synapse mask - connectivity/ inhibition/ excitation
        self.mask_method = mask_method
        self._conn_p, self._exc_p, self._inh_p, self._conn_n, self._exc_n, self._inh_n = \
            self._prep_inh_exc_conn_prop(inhibition_prop, excitation_prop, connectivity, 
                                         n=self.post_layer.size if mask_method == "by_pre" else \
                                            self.pre_layer.size if mask_method == "by_post" else \
                                            self.pre_layer.size * self.post_layer.size)
        # Mask for weights 
        self._weight_mask = np.ones((self.pre_layer.size, self.post_layer.size), dtype=np.int8)
        self._initialise_mask()

        # Self-retained pre- and post-neuron traces
        self._use_pre_trace = pre_trace
        if self._use_pre_trace:
            if self._step_wise:
                self._pre_trace = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
            elif self._event_driven:
                self._pretrace_tssp = np.full((self.pre_layer.size, self.post_layer.size), np.inf, dtype=np.float32)
                self._pretrace_last = 1.0
        self._use_post_trace = post_trace
        if self._use_post_trace:
            if self._step_wise:
                self._post_trace = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
            elif self._event_driven:
                self._posttrace_tssp = np.full((self.pre_layer.size, self.post_layer.size), np.inf, dtype=np.float32)
                self._posttrace_last = 1.0

        # Tau for pre-traces
        if tau_pre is not None and tau_pre > 1:
            tau_pre = tau_pre * dt
        self._tau_pre = tau_pre if tau_pre is not None else dt
        self._beta_pre = np.exp(-self.dt / self._tau_pre)
        # Tau for post-traces
        if tau_post is not None and tau_post > 1:
            tau_post = tau_post * dt
        self._tau_post = tau_post if tau_post is not None else dt
        self._beta_post = np.exp(-self.dt / self._tau_post)

        # Eligibility trace
        # Pre-before-post trace (previously just "eligibility_trace")
        self._use_elig_pre = eligibility_pre or eligibility_trace
        if self._use_elig_pre:
            if self._step_wise:
                self._etrace_pre = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
            elif self._event_driven:
                self._etssp_pre = np.full((self.pre_layer.size, self.post_layer.size), np.inf, dtype=np.float32)
                self._elast_pre = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
        # Post-before-pre trace
        self._use_elig_post = eligibility_post
        if self._use_elig_post:
            if self._step_wise:
                self._etrace_post = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
            elif self._event_driven:
                self._etssp_post = np.full((self.pre_layer.size, self.post_layer.size), np.inf, dtype=np.float32)
                self._elast_post = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
        # Combined STDP eligibility trace
        self._use_elig_stdp = eligibility_stdp
        self.ltd_coef = ltd_coef
        if self._use_elig_stdp:
            if self._step_wise:
                self._etrace_stdp = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
            elif self._event_driven:
                self._elast_stdp = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
                self._etssp_stdp = np.full((self.pre_layer.size, self.post_layer.size), np.inf, dtype=np.float32)
        # Custom Eligibility trace (for rule-controlled e-trace)
        self._use_elig_custom = eligibility_custom
        if self._use_elig_custom:
            # For the sake of simplicity, only step-wise method of updating custom eligibility will be used
            self._etrace_custom = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)

        
        # Constants for etraces
        if tau_syn is not None and isinstance(tau_syn, int):
            tau_syn = tau_syn * dt
        self._tau_syn = tau_syn if tau_syn is not None else dt
        self._beta_syn = np.exp(-self.dt / self._tau_syn)  # Decay rate for eligibility trace
        self._e_max = e_max if e_max is not None else np.inf
        self._e_min = e_min if e_min is not None else -np.inf

        # Weight decay
        self._decay_weight = False
        self._tau_weight = np.inf
        self._beta_weight = 1.0
        if tau_weight is not None:
            self._decay_weight = True
            if isinstance(tau_weight, int):
                tau_weight = tau_weight * dt
            self._tau_weight = tau_weight
            self._beta_weight = np.exp(-self.dt / self._tau_weight)
        
        # Initialize weights
        if weight_init not in ['uniform', 'normal', 'constant']:
            weight_init = 'uniform'
            Warning(f"Invalid weight initialisation method: {weight_init}. Using 'uniform' instead.")
        self.weight_init = weight_init
        self.weight_init_params = kwargs
        self._w_min = w_min if weight_clip_min is None else weight_clip_min
        self._w_max = w_max if weight_clip_max is None else weight_clip_max
        self.clip_weights = clip_weights
        self.normalise_weights = normalise_weights
        self.normalise_method = normalise_method
        # self.normalise_params = normalise_params if normalise_params is not None else {}
        self._init_weights()
        self._normalise_weights()

    def _prep_inh_exc_conn_prop(self, inhibition_prop: float|int, excitation_prop: float|int, connectivity: float, n: int):
        # (inh, exc, conn)
        if inhibition_prop is None:
            if excitation_prop is None:
                if connectivity is None:
                    # (-, -, -)
                    p_conn = 1.0
                    p_exc = 1.0
                    p_inh = 0.0
                    n_conn = n
                    n_exc = n
                    n_inh = 0
                else:
                    # (-, -, conn)
                    p_conn = np.clip(connectivity, 0, 1)
                    p_exc = p_conn
                    p_inh = 0.0
                    n_conn = round(n*p_conn)
                    n_exc = n_conn
                    n_inh = 0
            else:
                p_exc, n_exc = self._process_prop(excitation_prop, n)
                if connectivity is None:
                    # (-, exc, -)
                    p_conn = 1.0
                    p_inh = 1-p_exc
                    n_conn = n
                    n_inh = round(n*p_inh)
                else:
                    # (-, exc, conn)
                    p_conn = np.clip(connectivity, 0, 1)
                    n_conn = round(n*p_conn)
                    if n_exc > n_conn:
                        raise AssertionError(f"Number of excitatory connections {n_exc} cannot exceed number of possible connections {n_conn}")
                    p_inh = p_conn-p_exc
                    n_inh = round(n*p_inh)
        else:
            p_inh, n_inh = self._process_prop(inhibition_prop, n)
            if excitation_prop is None:
                if connectivity is None:
                    # (inh, -, -)
                    p_conn = 1.0
                    n_conn = n
                    p_exc = 1-p_inh
                    n_exc = round(p_exc)    
                else:
                    # (inh, -, conn)
                    p_conn = np.clip(connectivity, 0, 1)
                    n_conn = round(n*p_conn)
                    if n_inh > n_conn:
                        raise AssertionError(f"Number of inhibitory connections {n_inh} cannot exceed number of possible connections {n_conn}")
                    p_exc = p_conn-p_inh
                    n_exc = round(p_exc)    
            else:
                p_exc, n_exc = self._process_prop(excitation_prop, n)
                assert n_inh + n_exc <= n, f"Number of excitatory ({n_exc}) and inhibitory ({n_inh}) neurons cannot exceed total number of neurons {n}"
                if connectivity is None:
                    # (inh, exc, -)
                    p_conn = round(p_inh+p_exc, 2)
                    n_conn = n_inh+n_exc
                else:
                    # (inh, exc, conn)
                    p_conn = np.clip(connectivity, 0, 1)
                    n_conn = round(n*p_conn)
                    assert n_inh + n_exc == n_conn, f"Combined total of excitatory ({n_exc}) and inhibitory ({n_inh}) synapses must equal connectable synapses ({n_conn})"
        
        return p_conn, p_exc, p_inh, n_conn, n_exc, n_inh

    def _process_prop(self, prop: float|int, n):
        if isinstance(prop, int):
            assert prop <= n, f"Number of neurons {prop} must not be greater than total neurons {n}"
            n_ = prop
            p_ = n_ / n
        else:
            assert 0 <= prop <= 1, f"Proportion must be between 0 and 1. Got {prop}"
            p_ = float(prop)
            n_ = int(n*p_)
        return p_, n_

    def _initialise_mask(self):
        if self._inh_p == 0 and self._conn_p == 1.0:
            self._weight_mask.fill(1)
            return        
        self._weight_mask.fill(0)
        if self.mask_method == "random":
            # Probability applied to synapse layer as a whole
            idx = np.arange(self._weight_mask.size)
            np.random.shuffle(idx)
            self._weight_mask.flat[idx[:(self._exc_n)]] = 1
            self._weight_mask.flat[idx[(n-self._inh_n):]] = -1
            # if self._inh_p > 0:
            #     idx = np.random.binomial(1, p=self._inh_p, size=self._weight_mask.shape).astype(bool)
            #     self._weight_mask[idx] = -1
            # if self._conn_p < 1.0:
            #     idx = np.random.binomial(1, p=1-self._conn_p, size=self._weight_mask.shape).astype(bool)
            #     self._weight_mask[idx] = 0
        elif self.mask_method == "by_pre":
            # Each pre-neuron sends a fixed number of inhib/absent/excitatory synapses (-1/0/+1)
            n = self.post_layer.size
            # n_absent = np.round((1-self._conn_p) * self.post_layer.size).astype(int)
            # n_inhib = np.round(self._inh_p * self.post_layer.size).astype(int)
            for i in range(self.pre_layer.size):
                idx = np.arange(n)
                np.random.shuffle(idx)
                self._weight_mask[i, idx[:(self._exc_n)]] = 1
                self._weight_mask[i, idx[(n-self._inh_n):]] = -1
        elif self.mask_method == "by_post":
            # Each post-neuron receives a fixed number of inhib/absent/excitatory synapses (-1/0/+1)
            n = self.pre_layer.size
            # n_absent = np.round((1-self._conn_p) * self.pre_layer.size).astype(int)
            # n_inhib = np.round(self._inh_p * self.pre_layer.size).astype(int)
            for j in range(self.post_layer.size):
                idx = np.arange(n)
                np.random.shuffle(idx)
                self._weight_mask[idx[:(n-self._exc_n)], j] = 1
                self._weight_mask[idx[(n-self._inh_n):], j] = -1

    def _init_weights(self):
        if self.weight_init == 'uniform':
            wmin = self.weight_init_params.get('weight_init_min', 0.0)
            wmax = self.weight_init_params.get('weight_init_max', 1.0)
            self._weights = np.random.uniform(wmin, wmax, size=(self.pre_layer.size, self.post_layer.size))
        elif self.weight_init == 'normal':
            mean = self.weight_init_params.get('weight_init_mean', 0.0)
            std = self.weight_init_params.get('weight_init_std', 1.0)
            self._weights = np.random.normal(mean, std, size=(self.pre_layer.size, self.post_layer.size))
        elif self.weight_init == 'constant':
            value = self.weight_init_params.get('weight_init_value', 0.0)
            self._weights = np.full((self.pre_layer.size, self.post_layer.size), value, dtype=np.float32)
        else:
            raise NotImplementedError("Other weight initialisation methods not implemented yet")

    def _tile(self, vec_in: np.ndarray, vec_out: np.ndarray) -> np.ndarray:
        """
        Reshape input and output vectors to match weight matrix.
        """
        assert vec_in.shape[0] == self.pre_layer.size
        assert vec_out.shape[0] == self.post_layer.size
        vec_in = np.tile(vec_in, (self.weights.shape[1], 1)).T
        vec_out = np.tile(vec_out, (self.weights.shape[0], 1))
        return vec_in, vec_out
    
    def reset(self) -> None:
        """
        Reset the synaptic weights to their initial state.
        """
        self._init_weights()
        self._normalise_weights()
        self._initialise_mask()
        self.soft_reset()

    def soft_reset(self) -> None:
        """
        Reset only eligibility traces, but keep the synaptic weights unchanged.
        """
        if self._use_pre_trace:
            if self._step_wise:
                self._pre_trace.fill(0.0)
            elif self._event_driven:
                self._pretrace_tssp.fill(np.inf)
                # self._pretrace_last.fill(0.0)
        if self._use_post_trace:
            if self._step_wise:
                self._post_trace.fill(0.0)
            elif self._event_driven:
                self._posttrace_tssp.fill(np.inf)
                # self._posttrace_last.fill(0.0)
        if self._use_elig_pre:
            if self._step_wise:
                self._etrace_pre.fill(0.0)
            elif self._event_driven:
                self._etssp_pre.fill(np.inf)
                self._elast_pre.fill(0.0)
        if self._use_elig_post:
            if self._step_wise:
                self._etrace_post.fill(0.0)
            elif self._event_driven:
                self._etssp_post.fill(np.inf)
                self._elast_post.fill(0.0)
        if self._use_elig_stdp:
            if self._step_wise:
                self._etrace_stdp.fill(0.0)
            elif self._event_driven:
                self._etssp_stdp.fill(np.inf)
                self._elast_stdp.fill(0.0)
        if self._use_elig_custom:
            self._etrace_custom.fill(0.0)

        if self._apply_delay:
            self.current_buffer.reset()

    def forward(self, spike_input: np.ndarray) -> np.ndarray:
        """
        Compute the output current to the next neuron layer given a spike current input from the previous neuron layer.
        """
        assert spike_input.shape[0] == self.pre_layer.size
        assert spike_input.ndim == 1

        # Compute the output current
        output_current = np.dot(spike_input, self.weights * self._weight_mask)
        if self._apply_delay:
            output_current = self.current_buffer.push(output_current)
        return output_current

    def update(self) -> None:
        """
        Perform necessary updates of internal variables if applicable, including:
        - pre- and post-synaptic traces
        - eligibility traces (of various forms)
        - constant weight decay
        """
        self.update_traces()
        self.update_eligibility_trace()
        self.update_weight_decay()

    def update_weight_decay(self) -> None:
        if self._decay_weight:
            self._weights *= self._beta_weight
            # if new weights exceed boundaries, clip it back
            self._clip_weights()

    def update_traces(self) -> None:
        """Update pre- and post-synaptic traces, if enabled"""
        if self._use_pre_trace:
            if self._step_wise:
                self._pre_trace = self._pre_trace * self._beta_pre
                idx_spike = self.pre_layer.spike.nonzero()[0]
                self._pre_trace[idx_spike, :] = 1.0
            elif self._event_driven:
                self._pretrace_tssp += 1
                pre_spike = self.pre_layer.spike
                if any(pre_spike):
                    idx_spike = pre_spike.nonzero()[0]
                    self._pretrace_tssp[idx_spike, :] = 0
        if self._use_post_trace:
            if self._step_wise:
                self._post_trace *= self._beta_post
                idx_spike = self.post_layer.spike.nonzero()[0]
                self._post_trace[:, idx_spike] = 1.0
                # self._post_trace[:, :] = self.post_layer.spike[np.newaxis, :]
            elif self._event_driven:
                self._posttrace_tssp += 1
                post_spike = self.post_layer.spike
                if any(post_spike):
                    idx_spike = post_spike.nonzero()[0]
                    self._posttrace_tssp[:, idx_spike] = 0
    
    def update_eligibility_trace(self) -> None:
        # Gets called every timestep
        # Eligibility trace: pre-before-post
        if self._use_elig_pre:
            # Step-wise method:
            # Updates all synapses' traces based on decay rate + add a value rise when spike
            if self._step_wise:
                # self._update_etrace_step(self.post_layer, self.pre_layer, self._etrace_pre)
                post_spike = self.post_layer.spike
                if sum(post_spike) == 0:
                    rise = 0.0
                else:        
                    pre_trace = self.pre_layer.get_trace()
                    pre_trace, post_spike = self._tile(pre_trace, post_spike)
                    rise = pre_trace * post_spike
                # Insert ceiling operation here
                self._etrace_pre = np.clip(self._etrace_pre * self._beta_syn + rise, self.e_min, self.e_max)
            # Event-driven method:
            # Updates latest peak whenever new spike comes in. Actual trace value calculated when called.
            # (Actually more expensive since updating peaks require knowing current value and hence exponential calculation as well)
            elif self._event_driven:
                # self._update_etrace_event(self.post_layer, self.pre_layer, self._elast_pre, self._etssp_pre)
                # Always increment time since last peak for all synapses
                self._etssp_pre += 1
                post_spike = self.post_layer.spike
                # Check for incoming spike
                idx_spike = post_spike.nonzero()[0]
                if len(idx_spike) == 0:
                    # Skips entirely if no spikes
                    pass
                else:
                    # If at least 1 spike, find corresponding trace value of those spikes and update last peak value
                        # Value before rise (aka exponential decay)
                    decay = self._elast_pre[:, idx_spike] * np.exp(-self._etssp_pre[:, idx_spike] * self.dt / self._tau_syn) # Shape: [pre_size, num_post_spikes]
                        # Update last peak based on trace of pre-neuron
                    pre_trace = self.pre_layer.get_trace() # Shape: [pre_size,]
                    # Clipping operation
                    rise = pre_trace[:, np.newaxis]
                    self._elast_pre[:, idx_spike] = np.clip(rise + decay, self.e_min, self.e_max)
                        # Update tssp for post-neurons that spiked
                    self._etssp_pre[:, idx_spike] = 0
        # Eligibility trace: post-before-pre
        # Same with above but with pre and post roles reversed
        if self._use_elig_post:
            if self._step_wise:
                # self._update_etrace_step(self.pre_layer, self.post_layer, self._etrace_post)
                pre_spike = self.pre_layer.spike
                if sum(pre_spike) == 0:
                    rise = 0.0
                else:        
                    post_trace = self.post_layer.get_trace()
                    pre_spike, post_trace = self._tile(pre_spike, post_trace)
                    rise = post_trace * pre_spike
                self._etrace_post = np.clip(self._etrace_post * self._beta_syn + rise, self.e_min, self.e_max)
            elif self._event_driven:
                # self._update_etrace_event(self.pre_layer, self.post_layer, self._elast_post, self._etssp_post)
                self._etssp_post += 1
                pre_spike = self.pre_layer.spike
                idx_spike = pre_spike.nonzero()[0]
                if len(idx_spike) == 0:
                    pass
                else:
                        # Value before rise
                    decay = self._elast_post[idx_spike, :] * np.exp(-self._etssp_post[idx_spike, :] * self.dt / self._tau_syn) # Shape: [pre_size, num_post_spikes]
                        # Update last peak
                    post_trace = self.post_layer.get_trace() # Shape: [post_size,]
                    rise = post_trace[np.newaxis, :]
                    self._elast_post[idx_spike, :] = np.clip(rise + decay, self.e_min, self.e_max)
                        # Update tssp
                    self._etssp_post[idx_spike, :] = 0
        # Eligibility trace: combined STDP
        if self._use_elig_stdp:
            if self._step_wise:
                pre_spike = self.pre_layer.spike
                post_spike = self.post_layer.spike
                pre_spike, post_spike = self._tile(pre_spike, post_spike)
                if pre_spike.sum() == 0 and post_spike.sum() == 0:
                    rise = 0.0
                else:
                    post_trace = self.post_layer.get_trace()
                    pre_trace = self.pre_layer.get_trace()
                    pre_trace, post_trace = self._tile(pre_trace, post_trace)
                    rise = pre_trace * post_spike - self.ltd_coef * post_trace * pre_spike
                self._etrace_stdp = np.clip(self._etrace_stdp * self._beta_syn + rise, self.e_min, self.e_max)
            elif self._event_driven:
                self._etssp_stdp += 1
                pre_spike = self.pre_layer.spike
                post_spike = self.post_layer.spike            
                comb_spike = pre_spike | post_spike
                if comb_spike.sum() == 0:
                    pass
                else:
                    idx_spike = comb_spike.nonzero()

                    decay = self._elast_stdp[idx_spike] * np.exp(-self._etssp_stdp[idx_spike] * self.dt / self._tau_syn)

                    post_trace = self.post_layer.get_trace()
                    pre_trace = self.pre_layer.get_trace()

                    rise_ltp = np.where(post_spike, pre_trace, 0.0)
                    rise_ltd = np.where(pre_spike, post_trace, 0.0)

                    rise = rise_ltp - self.ltd_coef * rise_ltd

                    self._elast_stdp[idx_spike] = np.clip(decay + rise[idx_spike], self.e_min, self.e_max)

                    self._etssp_stdp[idx_spike] = 0
        # Custom eligiblity trace (for learning rule use only)
        if self._use_elig_custom:
            # only the decay part is updated here, the 'rise' part will be added when learning_rule is called
            self._etrace_custom = self._etrace_custom * self._beta_syn

    def get_pre_trace(self) -> np.ndarray | None:
        if self._use_pre_trace:
            if self._step_wise:
                return self._pre_trace
            else:
                return self._pretrace_last * np.exp(-self._pretrace_tssp * self.dt / self._tau_pre)
        else:
            pre_trace = self.pre_layer.get_trace()
            return np.broadcast_to(pre_trace[:, np.newaxis], (self.pre_layer.size, self.post_layer.size))

    def get_post_trace(self) -> np.ndarray | None:
        if self._use_post_trace:
            if self._step_wise:
                return self._post_trace
            else:
                return self._posttrace_last * np.exp(-self._posttrace_tssp * self.dt / self._tau_post)
        else:
            post_trace = self.post_layer.get_trace()
            return np.broadcast_to(post_trace[np.newaxis, :], (self.pre_layer.size, self.post_layer.size))

    def has_pre_trace(self) -> bool:
        return self._use_pre_trace

    def has_post_trace(self) -> bool:
        return self._use_post_trace

    def apply_learning_rule(self, reward: float = None, trigger_info: Dict[str, bool] = None) -> None:
        """
        Update the synaptic information by applying the stored learning rule.  

        Possible changes may include (depending on outputs of learning rule):
        - weight
        - threshold (of post-synaptic layer) 
        - eligibility trace (to 'eligibility_custom')
        """
        if self.plastic and self._learning_rule.check_trigger(trigger_info):
            self._apply_learning_rule(self._learning_rule, reward) 

    def apply_external_rule(self, reward: float = None, trigger_info: Dict[str, bool] = None) -> None:
        if self.plastic:
            if self._external_rule is not None and self._external_rule.check_trigger(trigger_info):
                self._apply_learning_rule(self._external_rule, reward) 

    def apply_internal_rule(self, trigger_info: Dict[str, bool] = None) -> None:
        if self.plastic:
            if self._internal_rule is not None and self._internal_rule.check_trigger(trigger_info):
                self._apply_learning_rule(self._internal_rule, reward=None) 

    def _apply_learning_rule(self, learning_rule: LearningRule, reward):
        dw, dth, delig = learning_rule.update(self, reward=reward, always_return_tuple=True)
        if dw is not None:
            self._update_weights(dw)
        if dth is not None: 
            self.post_layer.update_thresholds(dth)
        if delig is not None:
            self._etrace_custom = self._etrace_custom + delig
            self._etrace_custom = np.clip(self._etrace_custom, self.e_min, self.e_max)

    def update_weights_from_etrace(self, reward: float, etrace: Literal["pre", "post", "stdp", "custom"], lrate: float = 1.0) -> None:
        """
        Perform a fixed weight update, given reward and type of eligibility trace (must be enabled).

        Args:
            reward (float): Reward Signal
            etrace (Literal[&quot;pre&quot;, &quot;post&quot;, &quot;stdp&quot;, &quot;custom&quot;]): Type of eligibility trace to use
            lrate (float, optional): amplitude of weight change. Defaults to 1.0.
        """
        if self.plastic:
            dw = getattr(self, f"eligibility_{etrace}")
            dw = dw * reward
            self._update_weights(dw, lrate)

    def get_masked_weights(self) -> np.ndarray:
        return self._weights * self._weight_mask

    def _update_weights(self, dw, lrate: float = 1.0):
        self._weights += lrate * dw
        # Clip the weights
        self._clip_weights()
        # Normalise the weights
        self._normalise_weights()# Not sure if using np.clip or max(min()) is faster

    def _clip_weights(self):
        if self.clip_weights:
            self._weights = np.clip(self.weights, self.w_min, self.w_max)

    def _normalise_weights(self):
        if self.normalise_weights:
            self._weights = safe_norm(self.weights, self.normalise_method)

    def __repr__(self):
        return f"SynapseLayer({self.weights.shape})"

    @property
    def weights(self) -> np.ndarray:
        return self._weights
    @weights.setter
    def weights(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), "New weights must be an array"
        assert value.shape == self._weights.shape, f"Shape mismatch. Expected {self._weights.shape}, Got {value.shape}."
        self._weights = value

    @property
    def w_min(self) -> float:
        return self._w_min if self.clip_weights else -np.inf
    @w_min.setter
    def w_min(self, value):
        if self.clip_weights:
            if isinstance(value, np.ndarray):
                assert value.size == 1, "Input must either be a scalar or array of size 1"
                value = value.item()
            self._w_min = value
    @property
    def w_max(self) -> float:
        return self._w_max if self.clip_weights else np.inf
    @w_max.setter
    def w_max(self, value):
        if self.clip_weights:
            if isinstance(value, np.ndarray):
                assert value.size == 1, "Input must either be a scalar or array of size 1"
                value = value.item()
            self._w_max = value
    @property
    def w_range(self) -> float:
        return self.w_max - self.w_min
    @w_range.setter
    def w_range(self, value):
        if self.clip_weights:
            if isinstance(value, np.ndarray):
                assert value.size == 1, "Input must either be a scalar or array of size 1"
                value = value.item()
            assert value > 0, f"w_range must be strictly positive. Got value={value}"
            if not np.isinf(self.w_min):
                self.w_max = self.w_min + value
            else:
                warnings.warn(f"'w_min' is currently unbounded (value={self.w_min}). Cannot apply 'w_range' to create 'w_max'")

    @property
    def tau_weight(self) -> float:
        """
        Time constant for synaptic weight decay
        """
        return self._tau_weight
    @tau_weight.setter
    def tau_weight(self, value: float) -> None:
        if value is None:
            self._decay_weight = False
        else:
            if not self._decay_weight:
                self._decay_weight = True
            if isinstance(value, np.ndarray):
                assert value.size == 1, "Input must either be a scalar or array of size 1"
                value = value.item()
            assert value > 0, f"Time constant value must be strictly posive. Got tau_weight={value}"
            if isinstance(value, int) or value > 1:
                value *= self.dt
            self._tau_weight = value
            self._beta_weight = np.exp(-self.dt / self._tau_weight)

    @property
    def tau_syn(self) -> float:
        """
        Time constant used for all types of eligibility trace
        """
        return self._tau_syn
    @tau_syn.setter
    def tau_syn(self, value):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        assert value > 0, f"Time constant value must be strictly posive. Got tau_syn={value}"
        if isinstance(value, int) or value > 1:
            value *= self.dt
        self._tau_syn = value
        self._beta_syn = np.exp(-self.dt / self._tau_syn)

    @property
    def tau_pre(self) -> float:
        """
        Time constant used for pre-synaptic trace
        """
        return self._tau_pre
    @tau_pre.setter
    def tau_pre(self, value):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        assert value > 0, f"Time constant value must be strictly posive. Got tau_pre={value}"
        if isinstance(value, int) or value > 1:
            value *= self.dt
        self._tau_pre = value
        self._beta_pre = np.exp(-self.dt / self._tau_pre)

    @property
    def tau_post(self) -> float:
        """
        Time constant used for post-synaptic trace
        """
        return self._tau_post
    @tau_post.setter
    def tau_post(self, value):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        assert value > 0, f"Time constant value must be strictly posive. Got tau_post={value}"
        if isinstance(value, int) or value > 1:
            value *= self.dt
        self._tau_post = value
        self._beta_post = np.exp(-self.dt / self._tau_post)

    @property
    def e_min(self) -> float:
        return self._e_min
    @e_min.setter
    def e_min(self, value: float | np.ndarray):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        # assert value < self.e_max, f"New 'e_min' value must be strictly lower than current 'e_max'. Got e_min={value}"
        self._e_min = value

    @property
    def e_max(self) -> float:
        return self._e_max
    @e_max.setter
    def e_max(self, value: float):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        # assert value > self.e_min, f"New 'e_max' value must be strictly higher than current 'e_min'. Got e_max={value}"
        self._e_max = value

    @property
    def e_range(self) -> float:
        return self.e_max - self.e_min
    @e_range.setter
    def e_range(self, value: float):
        if isinstance(value, np.ndarray):
            assert value.size == 1, "Input must either be a scalar or array of size 1"
            value = value.item()
        assert value > 0, f"e_range must be strictly positive. Got e_range={value}"
        if not np.isinf(self.e_min):
            self.e_max = self.e_min + value
        else:
            warnings.warn(f"'e_min' is currently unbounded (e_min={self.e_min}). Cannot apply range to create 'e_max'")
    
    @property
    def eligibility_pre(self):
        """
        Calculate and return the Pre-before-Post eligibility trace if it is being used, otherwise return None.
        """
        if self._use_elig_pre:
            if self._step_wise:
                return self._etrace_pre
            elif self._event_driven:
                return self._elast_pre * np.exp(-self._etssp_pre * self.dt / self._tau_syn)
        else:
            return None

    @property
    def eligibility_post(self):
        """
        Calculate and return the Post-before-Pre eligibility trace if it is being used, otherwise return None.
        """
        if self._use_elig_post:
            if self._step_wise:
                return self._etrace_post
            elif self._event_driven:
                return self._elast_post * np.exp(-self._etssp_post * self.dt / self._tau_syn)
        else:
            return None
        
    @property
    def eligibility_stdp(self):
        """
        Calculate and return the combined STDP eligibility trace if it is being used, otherwise return None.
        """
        if self._use_elig_stdp:
            if self._step_wise:
                return self._etrace_stdp
            elif self._event_driven:
                return self._elast_stdp * np.exp(-self._etssp_stdp * self.dt / self._tau_syn)
        else:
            return None

    @property
    def eligibility_custom(self) -> np.ndarray:
        if self._use_elig_custom:
            return self._etrace_custom
        else:
            return None

    @property
    def learning_rule(self):
        return self._learning_rule
    
    @learning_rule.setter
    def learning_rule(self, rule: LearningRule):
        self._learning_rule = rule
        ## Fix: for STDP back-comp, external and internal rules may be specified at different times. Thus, cannot overwrite the other when not specified.
        # # Clear previous learning rule to avoid unexpected behaviours
        # self._internal_rule = None
        # self._external_rule = None
        # Extract internal and external learning Rule if possible
        if isinstance(rule, DualLearningRule): # type: ignore
            self._external_rule = rule.external_rule
            self._internal_rule = rule.internal_rule
        else:
            # Backward compatibility: 
            # # When single-rule is passed through, assign rule to internal/external rule based on trigger condtion
            if rule.trigger_condition == "on-timestep":
                self._internal_rule = rule
            elif rule.trigger_condition in ("on-step", "on-reward", "on-end"):
                self._external_rule = rule
            else:
                # When trigger condition is not set, do nothing
                pass
        # # Recreate custom eligibility trace if necessary
        # if self._out_eligibility and not self._use_elig_custom:
        #     self._use_elig_custom = True
        #     self._etrace_custom = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)
        # # Perform necessary changes based on rule encodings
        if isinstance(rule, EvolvableLearningRule):
            rule.apply_genes_to_synapse(self)

    def has_elig_pre(self):
        return self._use_elig_pre
    def has_elig_post(self):
        return self._use_elig_post
    def has_elig_stdp(self):
        return self._use_elig_stdp
    def has_elig_custom(self):
        return self._use_elig_custom


def safe_norm(array, method, params={}, eps=1e-10):
    if method == "sum":
        denom = np.sum(array, axis=0, keepdims=True)
    elif method == "L2":
        denom = np.linalg.norm(array, axis=0, keepdims=True)
    elif method == "P":
        p = params.get("p", 1)
        denom = np.linalg.norm(array, ord=p, axis=0, keepdims=True)
    else:
        raise ValueError(f"Normalisation method {method} not recognised.")
    
    # Avoid division by zero
    denom = np.where(denom == 0, eps, denom)
    array = array / denom
    return array

