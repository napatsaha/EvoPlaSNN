from typing import Literal

from common.base import LearningRule, NeuronLayerProtocol, SynapseLayerProtocol
import numpy as np
# from .neurons import NeuronLayer
from lrule import Empty_Rule
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
    
    def __init__(self, pre_layer: NeuronLayerProtocol, post_layer: NeuronLayerProtocol, *, 
                 learning_rule: LearningRule = None,
                 eligibility_trace: bool = False, eligibility_pre: bool = False, eligibility_post: bool = False,
                 eligibility_stdp: bool = False, ltd_coef: float = 1.0,
                 e_max: float = None, e_min: float = None,
                 tau_syn: float = None, dt: float = 1e-3,
                 synaptic_delay: int = 0,
                 sim_method: Literal["event-driven", "step-wise"] = "step-wise",
                 weight_init: Literal["uniform", "normal", "constant"] = "uniform",
                 weight_clip_min: float = 0.0, weight_clip_max: float = 1.0,
                 clip_weights: bool = True, normalise_weights: bool = False, 
                 normalise_method: Literal["sum", "L2", "P"] = "sum", #normalise_params: dict = None,
                 **kwargs):

        # Simulation parameters
        self.dt = dt
        self.sim_method = sim_method
        self._event_driven = sim_method == "event-driven"
        self._step_wise = sim_method == "step-wise"

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.learning_rule = learning_rule if learning_rule is not None else Empty_Rule()
        # self._learning_rule_type = self._get_lrule_type()

        # Synaptic Delay
        self.synaptic_delay = synaptic_delay
        self._apply_delay = synaptic_delay > 0
        if self._apply_delay:
            self.current_buffer = Array_FIFO(shape=(self.post_layer.size, ), size=synaptic_delay + 1)

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
        self._use_elig_custom = self._out_eligibility
        if self._use_elig_custom:
            # For the sake of simplicity, only step-wise method of updating custom eligibility will be used
            self._etrace_custom = np.zeros((self.pre_layer.size, self.post_layer.size), dtype=np.float32)

        
        # Constants for etraces
        if tau_syn is not None and isinstance(tau_syn, int):
            tau_syn = tau_syn * dt
        self._tau_syn = tau_syn if tau_syn is not None else dt
        self._beta_syn = np.exp(-self.dt / self._tau_syn)  # Decay rate for eligibility trace
        self.e_max = e_max if e_max is not None else np.inf
        self.e_min = e_min if e_min is not None else -np.inf
        
        # Initialize weights
        if weight_init not in ['uniform', 'normal', 'constant']:
            weight_init = 'uniform'
            Warning(f"Invalid weight initialisation method: {weight_init}. Using 'uniform' instead.")
        self.weight_init = weight_init
        self.weight_init_params = kwargs
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
        self.clip_weights = clip_weights
        self.normalise_weights = normalise_weights
        self.normalise_method = normalise_method
        # self.normalise_params = normalise_params if normalise_params is not None else {}
        self._init_weights()
        self._normalise_weights()

    def _init_weights(self):
        if self.weight_init == 'uniform':
            wmin = self.weight_init_params.get('weight_init_min', 0.0)
            wmax = self.weight_init_params.get('weight_init_max', 1.0)
            self.weights = np.random.uniform(wmin, wmax, size=(self.pre_layer.size, self.post_layer.size))
        elif self.weight_init == 'normal':
            mean = self.weight_init_params.get('weight_init_mean', 0.0)
            std = self.weight_init_params.get('weight_init_std', 1.0)
            self.weights = np.random.normal(mean, std, size=(self.pre_layer.size, self.post_layer.size))
        elif self.weight_init == 'constant':
            value = self.weight_init_params.get('weight_init_value', 0.0)
            self.weights = np.full((self.pre_layer.size, self.post_layer.size), value, dtype=np.float32)
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
        self.soft_reset()

    def soft_reset(self) -> None:
        """
        Reset only eligibility traces, but keep the synaptic weights unchanged.
        """
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
        output_current = np.dot(spike_input, self.weights)
        if self._apply_delay:
            output_current = self.current_buffer.push(output_current)
        return output_current
    
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

    # def _update_etrace_step(self, spike_layer: NeuronLayerProtocol, trace_layer: NeuronLayerProtocol, etrace):
    #     spike = spike_layer.spike
    #     if sum(spike) == 0:
    #         rise = 0.0
    #     else:        
    #         trace = trace_layer.get_trace()
    #         trace, spike = self._tile(trace, spike)
    #         rise = trace * spike
    #     etrace = etrace * self.beta_syn + rise

    # def _update_etrace_event(self, spike_layer: NeuronLayerProtocol, trace_layer: NeuronLayerProtocol, elast, etssp):
    #     etssp += 1
    #     spike = spike_layer.spike
    #     idx_spike = spike.nonzero()[0]
    #     if len(idx_spike) == 0:
    #         return
    #     else:
    #         trace = trace_layer.get_trace() # Shape: [pre_size,]
    #             # Value before rise
    #         decay = elast[:, idx_spike] * np.exp(-etssp[:, idx_spike] * self.dt / self.tau_syn) # Shape: [pre_size, num_post_spikes]
    #             # Update last peak
    #         elast[:, idx_spike] = trace[:, np.newaxis] + decay
    #             # Update tssp
    #         etssp[:, idx_spike] = 0

    def apply_learning_rule(self, reward: float = None) -> None:
        """
        Update the synaptic information by applying the stored learning rule.  

        Possible changes may include (depending on outputs of learning rule):
        - weight
        - threshold (of post-synaptic layer) 
        - eligibility trace (to 'eligibility_custom')
        """
        dw, dth, delig = self._learning_rule.update(self, reward=reward, always_return_tuple=True)
        if self._out_weights:
            self._update_weights(dw)
        if self._out_thresholds: 
            self.post_layer.update_thresholds(dth)
        if self._out_eligibility:
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
        dw = getattr(self, f"eligibility_{etrace}")
        dw = dw * reward
        self._update_weights(dw, lrate)

    def _update_weights(self, dw, lrate: float = 1.0):
        self.weights += lrate * dw
        # Clip the weights
        self._clip_weights()
        # Normalise the weights
        self._normalise_weights()# Not sure if using np.clip or max(min()) is faster

    def _clip_weights(self):
        if self.clip_weights:
            self.weights = np.clip(self.weights, self.weight_clip_min, self.weight_clip_max)

    def _normalise_weights(self):
        if self.normalise_weights:
            self.weights = safe_norm(self.weights, self.normalise_method)

    def __repr__(self):
        return f"SynapseLayer({self.weights.shape})"

    @property
    def tau_syn(self) -> float:
        """
        Time constant used for all types of eligibility trace
        """
        return self._tau_syn
    @tau_syn.setter
    def tau_syn(self, value):
        assert value > 0, f"Time constant value must be strictly posive. Got tau_syn={value}"
        self._tau_syn = value
        self._beta_syn = np.exp(-self.dt / self._tau_syn)
    
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
        self._out_weights = getattr(rule, "delta_weight", False)
        self._out_thresholds = getattr(rule, "delta_threshold", False)
        self._out_eligibility = getattr(rule, "delta_eligibility", False)
        # Perform necessary changes based on rule encodings
        if getattr(rule, "encode_tau_syn", False):
            value = getattr(rule, "values", {}).get("tau_syn")
            self.tau_syn = value

    def has_elig_pre(self):
        return self._use_elig_pre
    def has_elig_post(self):
        return self._use_elig_post
    def has_elig_stdp(self):
        return self._use_elig_stdp


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

