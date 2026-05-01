from typing import Callable, List, Literal
import yaml

import numpy as np

from common.utils import solve_hidden, calculate_size
from common.base import Genome
from genome.genome import BaseGenome, CompositeGenome
from .utils import tile_array
from .base import BaseLearningRule
from genome import parameter as param





def relu(x):
    """
    ReLU activation function.
    """
    return np.maximum(0, x)

def sigmoid(x):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Tanh activation function.
    """
    return np.tanh(x)

def linear(x):
    """
    Linear activation function.
    """
    return x

_activation_functions = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "linear": linear,
    "none": linear
}


class LinearLayer:
    """
    Simple fully-connected layer with weights and activation function.
    """
    def __init__(self, input_size, output_size, parameters=None, activation_function: str | Callable = None, bias=True, 
                 weight_dist: Literal["uniform", "normal"] = "uniform"):
        self.input_size = input_size
        self.output_size = output_size
        # self.parameters = parameters
        self._bias = bias
        self._init_weights(parameters, weight_dist)
        self.activation_function = _activation_functions.get(activation_function, linear) if not isinstance(activation_function, Callable) else activation_function


    def _init_weights(self, parameters: List | np.ndarray, weight_dist: str):
        if parameters is None:
            self.weight_dist = weight_dist
            self._randomise_weights()
        else:
            if isinstance(parameters, np.ndarray) and parameters.ndim > 1:
                parameters = parameters.flatten()
            # Decide if there is bias based on parameter length
            expected_size = self.input_size * self.output_size
            if len(parameters) == expected_size:
                self._bias = False
            elif len(parameters) == expected_size + self.output_size:
                self._bias = True
            else:
                # expected_size = self.input_size * (self.output_size + self._bias * self.output_size)
                raise ValueError(f"Expected parameters with either size {expected_size} or {expected_size + self.output_size}, got size {len(parameters)}")

            self.weight_dist = None
            self.weights = parameters[:(self.input_size * self.output_size)].reshape(self.input_size, self.output_size)
            if self._bias:
                self.bias = parameters[(-self.output_size):]
            else:
                self.bias = None

    def _randomise_weights(self):
        if self.weight_dist == "uniform":
            self.weights = np.random.rand(self.input_size, self.output_size)
            self.bias = np.random.rand(self.output_size) if self._bias else None
        elif self.weight_dist == "normal":
            self.weights = np.random.randn(self.input_size, self.output_size)
            self.bias = np.random.randn(self.output_size) if self._bias else None
        else:
            raise ValueError(f"Unknown weight distribution: {self.weight_dist}")
            
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the layer given the input x.
        """
        if self._bias:
            x = np.dot(x, self.weights) + self.bias
        else:
            x = np.dot(x, self.weights)
        return self.activation_function(x)
    
    @property
    def parameters(self):
        """
        Return the evolvable parameters of the layer.
        """
        if self._bias:
            return np.concatenate((self.weights.flatten(), self.bias))
        else:
            return self.weights.flatten()
        
    @parameters.setter
    def parameters(self, value: np.ndarray):
        """
        Set the parameters of the layer.
        """
        idx = self.input_size * self.output_size
        self.weights[:] = value[:idx].reshape(self.input_size, self.output_size)
        if self._bias:
            self.bias[:] = value[idx:]


    @property
    def size(self):
        return self.parameters.size

    def __repr__(self):
        return f"LinearLayer(input_size={self.input_size}, output_size={self.output_size}, activation_function={self.activation_function.__name__}, bias={self._bias})"
    
    def print_weights(self, precision: int = 6):
        s = ""
        prelim = "Weights: "
        bulk = str(self.weights.round(precision)).replace("\n", f"\n{len(prelim)*" "}")
        s += f"{prelim}{bulk}\n"
        if self._bias:
            prelim = "Bias: ".ljust(len(prelim)+1)
            bulk = str(self.bias.round(precision)).replace("\n", f"\n{len(prelim)*" "}")
            s += f"{prelim}{bulk}\n"
            # s += f"Bias:  {str(self.bias.round(precision)).replace("\n", "\n\t  ")}\n"
        s += f"Activation: {self.activation_function.__name__}"
        return s

    def __str__(self):
        return self.print_weights(precision=4)

class ANN:
    """
    A LearningRule that is approximated by a fully-connected ANN.
    """
    def __init__(self, input_size: int, hidden_size: List | int = None, output_size: int = 1, 
                 parameters: List = None, bias: bool = True,
                 hidden_activation: str | Callable = None, output_activation: str | Callable = None,
                 weight_dist: Literal["uniform", "normal"] = "uniform"):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_sizes = solve_hidden(hidden_size)
        self.output_size = int(output_size)
        self.hidden_activation = _activation_functions.get(hidden_activation, linear)
        self.output_activation = _activation_functions.get(output_activation, linear)
        self.weight_dist = weight_dist
        self.bias = bias
    
        self._create_layers(bias, parameters)

    # def _solve_hidden_sizes(self, hidden_size: List | int | None) -> List[int | None]:
    #     """
    #     Solve the hidden sizes for the ANN.
    #     """
    #     if hidden_size is None or hidden_size == 0 or len(hidden_size) == 0:
    #         return []
    #     elif isinstance(hidden_size, int):
    #         return [hidden_size]
    #     elif isinstance(hidden_size, list):
    #         return hidden_size
    #     else:
    #         raise ValueError("hidden_size must be an int or a list of ints")
        
    def _create_layers(self, bias, parameters=None):
        self.layers: List[LinearLayer] = []
        self.layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.num_layers = len(self.layer_sizes)
        
        if parameters is not None:
            target_size = calculate_size(self.input_size, self.hidden_sizes, self.output_size, bias)
            if len(parameters) != target_size:
                raise ValueError(f"Parameters must have size {target_size}. Got {len(parameters)} instead.")
            current_index = 0
        for i in range(self.num_layers - 1):
            if parameters is not None: 
                expected_size = self.layer_sizes[i] * self.layer_sizes[i + 1] + bool(bias) * self.layer_sizes[i + 1]
                params = parameters[current_index:current_index + expected_size]
                current_index += expected_size
            else:
                params = None
            layer = LinearLayer(self.layer_sizes[i], self.layer_sizes[i + 1], 
                                activation_function=self.hidden_activation if i < self.num_layers - 2 else self.output_activation,
                                bias=bias, parameters=params, weight_dist=self.weight_dist)
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Feedforward.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
        # dw = self.forward(x)
        # if w is None:
        #     return dw
        # else:
        #     return w + dw

    @property
    def parameters(self):
        """
        Return the evolvable parameters of the ANN.
        """
        params = []
        for layer in self.layers:
            params.append(layer.parameters)
        return np.concatenate(params)
    
    @parameters.setter
    def parameters(self, value: np.ndarray):
        """
        Set the parameters of the ANN.
        """
        if len(value) != self.size:
            raise ValueError(f"Parameters must have size {self.size}. Got {len(value)} instead.")
        current_index = 0
        for layer in self.layers:
            expected_size = layer.size
            layer.parameters = value[current_index:current_index + expected_size]
            current_index += expected_size
    

    @property
    def weights(self):
        return [layer.weights for layer in self.layers]
    
    @property
    def biases(self):
        return [layer.bias for layer in self.layers]

    @property
    def size(self):
        return sum([layer.size for layer in self.layers])

    def __repr__(self):
        return f"ANN(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, output_size={self.output_size}, bias={self.bias}, " + \
                f"hidden_activation={self.hidden_activation.__name__}, output_activation={self.output_activation.__name__})"
    
    def __str__(self):
        s = "{\n"
        for layer in self.layers:
            s += "  " + str(layer) + "\n"
        s += "}"
        return s


class ANN_Rule(BaseLearningRule):
    """
    A Learning Rule that represents a black box ANN function that converts synapse-related information to weight updates.
    """
    INPUT_ORDER = ("trace_pre", "trace_post", "weights", "reward", "eligibility_pre", "eligibility_post", "eligibility_stdp")
    OUTPUT_ORDER = ("weight", "threshold")
    AGG_DICT = {
                "max": np.max,
                "min": np.min,
                "mean": np.mean,
                "sum": np.sum
                }

    genome: Genome

    def __init__(self, parameters=None, *, 
                 encode_learning_rate: bool = False,
                 encode_hidden_activation: bool = False, 
                 encode_output_activation: bool = False,
                 # ANN params
                 hidden_size: List | int = None, bias: bool = True,
                 hidden_activation: str = None, output_activation: str = None,
                 weight_dist: Literal["uniform", "normal"] = "uniform",
                 # LearningRule params
                 learning_rate: float = 1.0, learning_rate_thr: float = 0.1, threshold_agg_func: Literal["max", "min", "mean", "sum"] = "mean",
                 delta_weight: bool = True, delta_threshold: bool = False,
                 use_trace_pre: bool = False, use_trace_post: bool = False, use_weights: bool = True, use_reward: bool = False, 
                 use_eligibility: bool = False, use_eligibility_pre: bool = False, use_eligibility_post: bool = False, use_eligibility_stdp: bool = False,
                 **kwargs):
        # If there is parameters, need to figure out which values belong to which gene first
        _later = False # Tag to tell whether to do something later after constructing BaseLearningRule and ANN
        if parameters is not None:
            _later = True
            i = 0 # Counting index from the back of parameters array for each gene to be encoded
            # Assuming a gene order of [weight, learning rate]
            if encode_learning_rate:
                i += 1
                val = parameters[-1]
                learning_rate_gene = param.UniformBounded(val, low=0, high=1) # Placeholder type
                learning_rate = learning_rate_gene.value # Overwrite init argument
            # Finally, access weight values from parameters
            val = parameters[:-i] if i > 0 else parameters
            weights_gene = param.UniformBoundedArray(val, low=0, high=1) # Again placeholder type
            weights = weights_gene.value
        else:
            weights = None

        # # Allocate genome and ann parameters properly
        # if encode_learning_rate:
        #     if parameters is not None:
        #         learning_rate = parameters[-1]
        #         ann_parameters = parameters[:-1]
        #     else:
        #         # ann_parameters = param.UniformBoundedArray(parameters, size=len(parameters), low=-1, high=1).value
        #         ann_parameters = None
        #         # learning_rate = param.UniformBounded(low=0, high=1).value
        #         learning_rate = np.random.uniform(0, 1)
        # else:
        #     ann_parameters = parameters

        BaseLearningRule.__init__(self, learning_rate=learning_rate, learning_rate_thr=learning_rate_thr, threshold_agg_func=threshold_agg_func, 
                        delta_weight=delta_weight, delta_threshold=delta_threshold, 
                        use_trace_pre=use_trace_pre, use_trace_post=use_trace_post, use_weights=use_weights, use_reward=use_reward, 
                        use_eligibility=use_eligibility, use_eligibility_pre=use_eligibility_pre, use_eligibility_post=use_eligibility_post, use_eligibility_stdp=use_eligibility_stdp, 
                        **kwargs)

        # Construct an ANN
        self.ann = ANN(input_size=self.input_size, output_size=self.output_size, parameters=weights, 
                       hidden_size=hidden_size, hidden_activation=hidden_activation, output_activation=output_activation,
                       bias=bias, weight_dist=weight_dist,
                       **kwargs)
        
        if _later:
            genes = []
            genes.append(weights_gene)
            if encode_learning_rate:
                genes.append(learning_rate_gene)
        else: # If not constructing genes earlier because no parameter is passed through
            genes = []
            val = self.ann.parameters
            weights_gene = param.UniformBoundedArray(val, low=0, high=1) # Placeholder type
            genes.append(weights_gene)
            if encode_learning_rate:
                val = self.learning_rate
                learning_rate_gene = param.UniformBounded(val, low=0, high=1)
                genes.append(learning_rate_gene)


        # genome_parameters = np.r_[self.ann.parameters, self.learning_rate] if encode_learning_rate else self.ann.parameters
        self.genome = CompositeGenome(genes=genes)

        # self.weight_dist = self.ann.weight_dist


    def forward(self, inp):
        return self.ann.forward(inp)

    def mutate(self, rate: float) -> np.ndarray:
        genome = self.parameters.copy()
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        gene_to_mutate = np.random.randint(self.size, size=(int(rate*self.size), ))
        for gene_id in gene_to_mutate:
            if self.ann.weight_dist == "uniform":
                genome[gene_id] = np.random.rand()
            elif self.ann.weight_dist == "normal":
                genome[gene_id] = np.random.randn()
        return genome
        # return self.__class__(parameters=genome, **self.__dict__)


    def save_parameters(self, file_path: str, precision: int = 6):
        """
        Save only flattened parameters of the ANN Rule to a file. (Intended to use with `lrule.ann.read_ANN_Rule()` function)
        """
        np.savetxt(file_path, self.ann.parameters, delimiter=',', fmt=f'%.{precision}f',)

    def save_to_file(self, file_path: str):
        """
        Save the ANN Rule as a numpy zip .npz file, containing all meta information about the rule. 
        (Intended to use with `lrule.ann.ANN_Rule.load_from_file()` method)
        """
        np.savez(file_path, parameters=self.ann.parameters, input_size=self.input_size,
                 use_trace_pre=self.use_trace_pre, use_trace_post=self.use_trace_post,
                 use_weights=self.use_weights, use_reward=self.use_reward, 
                 use_eligilibity_pre=self.use_eligibility_pre, use_eligibility_post=self.use_eligibility_post,
                 use_eligibility_stdp=self.use_eligibility_stdp,
                 hidden_size=self.ann.hidden_sizes, bias=self.ann.bias,
                 hidden_activation=self.ann.hidden_activation.__name__,
                 output_activation=self.ann.output_activation.__name__,)
        
    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load the ANN Rule from a .npz file.
        (Intended to use with `lrule.ann.ANN_Rule.save_to_file()` method)
        """
        data = np.load(file_path)
        parameters = data['parameters']
        use_trace_pre = data['use_trace_pre'].item()
        use_trace_post = data['use_trace_post'].item()
        use_weights = data['use_weights'].item()
        use_reward = data['use_reward'].item()
        use_eligibility = data.get('use_eligibility', False).item()  # Optional, defaults to False if not present
        use_eligibility_pre = data.get('use_eligibility_pre', False).item()
        use_eligibility_post = data.get('use_eligibility_post', False).item()
        use_eligibility_stdp = data.get('use_eligibility_stdp', False).item()
        
        return cls(parameters=parameters,
                   use_trace_pre=use_trace_pre, 
                   use_trace_post=use_trace_post,
                   use_weights=use_weights, 
                   use_reward=use_reward,
                   use_eligibility_pre=use_eligibility_pre or use_eligibility, 
                   use_eligibility_post=use_eligibility_post,
                   use_eligibility_stdp=use_eligibility_stdp,
                   hidden_size=data["hidden_size"].tolist(),
                   bias=data["bias"].item(),
                   hidden_activation=data["hidden_activation"].item(),
                   output_activation=data["output_activation"].item())

    @property
    def size(self):
        return self.genome.size
    
    @property
    def parameters(self):
        return self.genome.parameters
    
    @parameters.setter
    def parameters(self, value):
        # TODO: Edit for multiple gene types
        self.ann.parameters = value
        self.genome.parameters = value

    def __repr__(self):
        # return f"ANN_Rule(parameters_size={self.size}, use_trace_pre={self.use_trace_pre}, use_trace_post={self.use_trace_post}, use_weights={self.use_weights}, use_reward={self.use_reward}, " + \
        # f"hidden_size={self.ann.hidden_sizes}, bias={self.ann.bias})"
        return f"ANN_Rule(parameters={self.parameters.round(2)}, size={self.size}, inputs={self.input_order}, outputs={self.output_order}, " + \
            f"learning_rate={[self.learning_rate, self.learning_rate_thr]})"
    
    def __str__(self):
        s = "ANN_Rule("
        s += "Inputs: "
        for i, layer in enumerate(self.ann.layers):
            s += f"\n  Layer {i}: (\n    "
            s += layer.print_weights(precision=4).replace("\n", "\n    ")
            s +=  "\n  ),"
        s += "\n)"
        return s
    

def read_ANN_Rule(parameter_path: str, config_path: str) -> ANN_Rule:
    """
    Create an ANN Rule from a file with parameters and configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parameters = np.loadtxt(parameter_path, delimiter=',')
    if "arule_params" in config:
        config["lrule_params"] = config["arule_params"]
    if "type" in config["lrule_params"]:
        lrule_type = config["lrule_params"].pop("type")
        if lrule_type != "ann":
            raise ValueError("Only lrule_params with type='ann' can be used to construct an ANN Rule.")
    return ANN_Rule(parameters=parameters, **config["lrule_params"])