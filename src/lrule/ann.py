from typing import Any, Callable, Dict, Iterable, List, Literal, Sequence, Tuple
import yaml

import numpy as np
from numpy.typing import ArrayLike

from common.utils import solve_hidden, calculate_size
from common.base import Genome, Parameter
from genome.genome import CompositeGenome, EvolvableLearningRule
from genome import parameter as param

from .base import BaseLearningRule





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


def _process_activation_func(value, value_name: str = "Value") -> Tuple[str, Callable]:
    if value is None:
        name = "none"
        func = _activation_functions.get(name)
    elif isinstance(value, str):
        name = value
        assert name in _activation_functions, f"Function {name} must be in supported activation functions: {_activation_functions.keys()}"
        func = _activation_functions.get(value, linear)
    elif isinstance(value, Callable):
        name = getattr(value, "__name__")
        func = value
    elif isinstance(value, Iterable):
        name, func = [], []
        for val in value:
            n, f = _process_activation_func(val, value_name)
            name.append(n)
            func.append(f)
    else:
        raise TypeError(f"{value_name} must be of type 'str' or 'Callable'. Got {type(value)}")
    return name, func


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
                 hidden_activation: str | Callable = "linear", output_activation: str | Callable = "linear",
                 weight_dist: Literal["uniform", "normal"] = "uniform"):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_sizes = solve_hidden(hidden_size)
        self.num_hidden = len(self.hidden_sizes)
        self.output_size = int(output_size)
        self.hidden_activation, self._hidden_func = _process_activation_func(hidden_activation, "hidden_activation")
        self.output_activation, self._output_func = _process_activation_func(output_activation, "output_activation")
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
            if i < self.num_layers - 2:
                func = self._hidden_func[i] if hasattr(self._hidden_func, "__getitem__") else self._hidden_func
            else:
                func = self._output_func
            layer = LinearLayer(self.layer_sizes[i], self.layer_sizes[i + 1], 
                                activation_function=func,
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
                f"hidden_activation={self.hidden_activation}, output_activation={self.output_activation})"
    
    def __str__(self):
        s = "{\n"
        for layer in self.layers:
            s += "  " + str(layer) + "\n"
        s += "}"
        return s


class ANN_Rule(BaseLearningRule, EvolvableLearningRule):
    """
    A Learning Rule that represents a black box ANN function that converts synapse-related information to weight updates.
    """
    # INPUT_ORDER = ("trace_pre", "trace_post", "weights", "reward", "eligibility_pre", "eligibility_post", "eligibility_stdp")
    # OUTPUT_ORDER = ("weight", "threshold")
    # AGG_DICT = {
    #             "max": np.max,
    #             "min": np.min,
    #             "mean": np.mean,
    #             "sum": np.sum
    #             }
    GENE_ORDER = ("weights", "learning_rate", "hidden_activation", "output_activation")
    ACTIVATION_FUNC_ORDER = ("linear", "relu", "sigmoid", "tanh")
    default_gene_order = ("weights", "learning_rate", "hidden_activation", "output_activation", "tau_syn", )

    genome: Genome


    def __init__(self, parameters: ArrayLike = None, genes: List[Parameter] = None, 
                 genes_to_encode: List[Dict] | Dict[str, Dict] = None, gene_order: Sequence[str] = None, *, 
                 # Which gene to encode
                 encode_learning_rate: bool = None,
                 encode_hidden_activation: bool = None, 
                 encode_output_activation: bool = None,
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
        
        BaseLearningRule.__init__(self, learning_rate=learning_rate, learning_rate_thr=learning_rate_thr, threshold_agg_func=threshold_agg_func, 
                        delta_weight=delta_weight, delta_threshold=delta_threshold, 
                        use_trace_pre=use_trace_pre, use_trace_post=use_trace_post, use_weights=use_weights, use_reward=use_reward, 
                        use_eligibility=use_eligibility, use_eligibility_pre=use_eligibility_pre, 
                        use_eligibility_post=use_eligibility_post, use_eligibility_stdp=use_eligibility_stdp, 
                        **kwargs)

        self._weight_size = calculate_size(self.input_size, hidden_size, self.output_size, bias)
        self._num_hidden_layers = len(solve_hidden(hidden_size))
        if self._num_hidden_layers < 1:
            encode_hidden_activation = False

        if gene_order is None and genes_to_encode is None:
            encodings = {"hidden_activation": encode_hidden_activation,
                         "output_activation": encode_output_activation,
                         "learning_rate": encode_learning_rate,
                         "weights": True}
            gene_order = [gene for gene in self.GENE_ORDER if encodings.get(gene)]

        EvolvableLearningRule.__init__(self, parameters=parameters, genes=genes, genes_to_encode=genes_to_encode, gene_order=gene_order)

        self.ann = ANN(input_size=self.input_size, output_size=self.output_size, parameters=self._weights, 
                    hidden_size=hidden_size, 
                    hidden_activation=self._hidden_activation if self._hidden_activation is not None else hidden_activation, 
                    output_activation=self._output_activation if self._output_activation is not None else output_activation,
                    bias=bias, weight_dist=self._weight_dist if self._weight_dist is not None else weight_dist,
                    **kwargs)

        # self.encode_weights = True # Weights is always encoded
        # self.encode_learning_rate = encode_learning_rate
        # self.encode_hidden_activation = encode_hidden_activation
        # self.encode_output_activation = encode_output_activation

        # self.genes_to_encode = genes_to_encode
        # self._genes_params = {}
        # # TODO: Verify that list of encodings passed through is compatible with as self.encodings
        # if self.genes_to_encode is not None:
        #     # Convert to easier format of Dict[name: Dict[params]]
        #     self._genes_params = {}
        #     for i, g_dict in enumerate(self.genes_to_encode):
        #         g_dict = g_dict.copy()
        #         name = g_dict.pop("name")
        #         if name is None:
        #             raise ValueError(f"Name must be defined in entry {i} of genes_to_encode")
        #         self._genes_params[name] = g_dict

        # # Prevents a case where both parameters and genes are passed through
        # if (parameters is not None) and (genes is not None):
        #     raise ValueError("Only one of 'parameters' or 'genes' can be passed, not both.")

        # weight_size = calculate_size(self.input_size, hidden_size, self.output_size, bias)
        # num_hidden_layers = len(solve_hidden(hidden_size))
        # if num_hidden_layers < 1:
        #     self.encode_hidden_activation = False
            
        # # TODO: Ensure ordering matches self.GENE_ORDER
        # self.encodings = [self.encode_weights, self.encode_learning_rate, self.encode_hidden_activation, self.encode_output_activation]

        # # CASE 1: An array of parameters is passed through
        # # Create genes from parameters according to gene_params (if present)
        # if parameters is not None:
        #     genes = []
        #     i = 0
        #     for enc_flag, enc_type in zip(self.encodings, self.GENE_ORDER):
        #         if not enc_flag:
        #             continue
        #         if enc_type == "weights":
        #             # Extract positions from full genome
        #             val = parameters[0:weight_size]
        #             if enc_type in self._genes_params:
        #                 gene_params = self._genes_params.get(enc_type).copy()
        #                 kind = gene_params.pop("kind")
        #                 gene = param.create_param(kind=kind, length=weight_size, value=val, **gene_params)
        #             else:
        #                 gene = param.RealParam(value=val, length=weight_size, dist=weight_dist)
        #             weights = gene.value
        #             weight_dist = getattr(gene, "dist", None)
        #             i += gene.length
        #             genes.append(gene)
        #         elif enc_type == "learning_rate":
        #             val = parameters[i:(i+1)]
        #             if enc_type in self._genes_params:
        #                 gene_params = self._genes_params.get(enc_type).copy()
        #                 kind = gene_params.pop("kind")
        #                 gene = param.create_param(kind=kind, length=1, value=val, **gene_params)
        #             else:
        #                 gene = param.RealParam(value=val, dist="normal", low=0)
        #             learning_rate = gene.value
        #             i += gene.length
        #             genes.append(gene)
        #         elif enc_type == "hidden_activation":
        #             val = parameters[i:(i+num_hidden_layers)]
        #             if enc_type in self._genes_params:
        #                 gene_params = self._genes_params.get(enc_type).copy()
        #                 kind = gene_params.pop("kind")
        #                 gene = param.create_param(kind=kind, length=num_hidden_layers, value=val, **gene_params)
        #             else:
        #                 gene = param.DiscreteParam(value=val, length=num_hidden_layers, n=len(self.GENE_ORDER))
        #             hidden_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value)
        #             i += gene.length
        #             genes.append(gene)
        #         elif enc_type == "output_activation":
        #             val = parameters[i:(i+1)]
        #             if enc_type in self._genes_params:
        #                 gene_params = self._genes_params.get(enc_type).copy()
        #                 kind = gene_params.pop("kind")
        #                 gene = param.create_param(kind=kind, length=1, value=val, **gene_params)
        #             else:
        #                 gene = param.DiscreteParam(value=val, length=1, n=len(self.GENE_ORDER))
        #             output_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value).item() # Enforce scalar value
        #             i += gene.length
        #             genes.append(gene)
        #         else:
        #             raise NotImplementedError(f"Encoding for {enc_type} not supported yet.")

        # # CASE 2: A list of genes is passed through
        # # Extract values from passed-in genes
        # elif genes is not None:
        #     assert isinstance(genes, List), "Genes must be a list of parameters"
        #     assert len(genes) == sum(self.encodings), "Length of gene objects must equal number of enabled encoding"
        #     i = 0
        #     for enc_flag, enc_type in zip(self.encodings, self.GENE_ORDER):
        #         if not enc_flag:
        #             continue
        #         if enc_type == "weights":
        #             weights = genes[i].value
        #             weight_dist = getattr(genes[i], "dist", None)
        #             assert len(weights) == weight_size, f"Values of {enc_type} must be of length {weight_size}. Got {len(weights)}"
        #             i += 1
        #         elif enc_type == "learning_rate":
        #             learning_rate = genes[i].value
        #             assert len(learning_rate) == 1, f"Values of {enc_type} must be of length {1}. Got {len(learning_rate)}"
        #             i += 1
        #         elif enc_type == "hidden_activation":
        #             val = genes[i].value
        #             assert len(val) == num_hidden_layers, f"Values of {enc_type} must be of length {num_hidden_layers}. Got {len(val)}"
        #             hidden_activation = np.take(self.ACTIVATION_FUNC_ORDER, val)
        #             i += 1
        #         elif enc_type == "output_activation":
        #             val = genes[i].value
        #             assert len(val) == 1, f"Values of {enc_type} must be of length {1}. Got {len(val)}"
        #             output_activation = np.take(self.ACTIVATION_FUNC_ORDER, val).item() # Enforce scalar value
        #             i += 1
        #         else:
        #             raise NotImplementedError(f"Encoding for {enc_type} not supported yet.")
        
        # # CASE 3: If `genes_to_encode` instruction is given for how to generate values for encoded genes
        # # Create random genes from passed-in instructions
        # elif genes_to_encode is not None:
        #     genes = []
        #     for enc_flag, enc_type in zip(self.encodings, self.GENE_ORDER):
        #         if not enc_flag:
        #             continue
        #         if enc_type == "weights":
        #             gene_params = self._genes_params.get(enc_type).copy()
        #             kind = gene_params.pop("kind")
        #             gene = param.create_param(kind=kind, length=weight_size, **gene_params)
        #             weight_dist = getattr(gene, "dist", None)
        #             weights = gene.value
        #             genes.append(gene)
        #         elif enc_type == "learning_rate":
        #             gene_params = self._genes_params.get(enc_type).copy()
        #             kind = gene_params.pop("kind")
        #             gene = param.create_param(kind=kind, length=1, **gene_params)
        #             learning_rate = gene.value
        #             genes.append(gene)
        #         elif enc_type == "hidden_activation":
        #             gene_params = self._genes_params.get(enc_type).copy()
        #             kind = gene_params.pop("kind")
        #             gene = param.create_param(kind=kind, length=num_hidden_layers, **gene_params)
        #             hidden_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value)
        #             genes.append(gene)
        #         elif enc_type == "output_activation":
        #             gene_params = self._genes_params.get(enc_type).copy()
        #             kind = gene_params.pop("kind")
        #             gene = param.create_param(kind=kind, length=1, **gene_params)
        #             output_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value).item() # Enforce scalar value
        #             genes.append(gene)
        #         else:
        #             raise NotImplementedError(f"Encoding for {enc_type} not supported yet.")

        # # CASE 0: Nothing is passed through (but genes must be created accordingly)
        # # Create random genes based on default Parameter classes for each encoding
        # else:
        #     genes = []
        #     # weights = None
        #     for enc_flag, enc_type in zip(self.encodings, self.GENE_ORDER):
        #         if not enc_flag:
        #             continue
        #         if enc_type == "weights":
        #             gene = param.RealParam(length=weight_size, dist=weight_dist, low=0, high=1)
        #             weights = gene.value
        #             genes.append(gene)
        #         elif enc_type == "learning_rate":
        #             gene = param.RealParam(length=1, dist="uniform", low=0, high=1)
        #             learning_rate = gene.value
        #             genes.append(gene)
        #         elif enc_type == "hidden_activation":
        #             gene = param.DiscreteParam(length=num_hidden_layers, n=len(self.ACTIVATION_FUNC_ORDER))
        #             hidden_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value)
        #             genes.append(gene)
        #         elif enc_type == "output_activation":
        #             gene = param.DiscreteParam(length=1, n=len(self.ACTIVATION_FUNC_ORDER))
        #             output_activation = np.take(self.ACTIVATION_FUNC_ORDER, gene.value).item() # Enforce scalar value
        #             genes.append(gene)     
        #         else:
        #             raise NotImplementedError(f"Encoding for {enc_type} not supported yet.")

        # # Update learning rate (in case value is pulled from genome)
        # self.learning_rate = learning_rate

        # # Construct genome from stored genes
        # if genes is None:
        #     raise NameError("Variable genes must be defined")
        # self.genome = CompositeGenome(genes=genes)

        # Construct an ANN
        # self.ann = ANN(input_size=self.input_size, output_size=self.output_size, parameters=weights, 
        #                hidden_size=hidden_size, hidden_activation=hidden_activation, output_activation=output_activation,
        #                bias=bias, weight_dist=weight_dist,
        #                **kwargs)
        
    def _build_gene_specs(self) -> Dict[str, Dict[str, Any]]:
        specs = super()._build_gene_specs()
        specs.update({
            "weights": dict(kind="real", length=self._weight_size, dist="uniform", dist_params=dict(low=0, high=1)),
            "hidden_activation": dict(kind="discrete", length=self._num_hidden_layers, n=len(self.ACTIVATION_FUNC_ORDER)),
            "output_activation": dict(kind="discrete", length=1, n=len(self.ACTIVATION_FUNC_ORDER))
        })
        return specs

    def _apply_gene_values(self):
        super()._apply_gene_values()
        # Extract weight information
        if self.encode_weights:
            self._weights = self.values.get("weights", None)
            self._weight_dist = [getattr(gene, "dist") for gene, name in zip(self.genes, self.gene_order) if name == "weights"][0]
        else:
            self._weights = None
            self._weight_dist = None
        # Extract activation function information
        if self.encode_hidden_activation:
            self._hidden_activation = np.take(self.ACTIVATION_FUNC_ORDER, self.values.get("hidden_activation"))
        else:
            self._hidden_activation = None
        if self.encode_output_activation:
            self._output_activation = np.take(self.ACTIVATION_FUNC_ORDER, self.values.get("output_activation")).item()
        else:
            self._output_activation = None

    def forward(self, inp):
        return self.ann.forward(inp)

    def mutate(self, rate: float = 1.0, scale: float = 0.1, method: Literal["resample", "perturb"] = "resample") -> 'ANN_Rule':
        new_genes = self.genome.mutate(rate=rate, scale=scale, method=method, return_genes_only=True)
        return self.__class__(genes = new_genes, **self.to_dict())

    def crossover(self, other: 'ANN_Rule', rate: float = 0.5) -> 'ANN_Rule':
        new_genes = self.genome.crossover(other.genome, rate, return_genes_only=True)
        return self.__class__(genes = new_genes, **self.to_dict())

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(dict(
            # Encoding params
            gene_order = self.gene_order,
            # ANN params
            hidden_activation = self.ann.hidden_activation,
            output_activation = self.ann.output_activation,
            weight_dist = self.ann.weight_dist,
            bias = self.ann.bias,
            hidden_size = self.ann.hidden_sizes,
        ))
        return d

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

    @property
    def encode_weights(self) -> bool:
        return "weights" in self._gene_order

    @property
    def encode_hidden_activation(self) -> bool:
        return "hidden_activation" in self._gene_order

    @property
    def encode_output_activation(self) -> bool:
        return "output_activation" in self._gene_order

    def __repr__(self):
        # return f"ANN_Rule(parameters_size={self.size}, use_trace_pre={self.use_trace_pre}, use_trace_post={self.use_trace_post}, use_weights={self.use_weights}, use_reward={self.use_reward}, " + \
        # f"hidden_size={self.ann.hidden_sizes}, bias={self.ann.bias})"
        return f"ANN_Rule(parameters={self.parameters.round(2)}, size={self.size}, inputs={self.input_order}, outputs={self.output_order}, " + \
            f"learning_rate={[self.learning_rate, self.learning_rate_thr]})"
    
    def __str__(self):
        # TODO: Print Learning Rate as well
        s = "ANN_Rule("
        s += f"\n  Learning Rate: {self.learning_rate}"
        s += f"\n  Inputs: {self.input_order}"
        for i, layer in enumerate(self.ann.layers):
            s += f"\n  Layer {i}: (\n    "
            s += layer.print_weights(precision=4).replace("\n", "\n    ")
            s +=  "\n  ),"
        s += f"\n  Outputs: {self.output_order}"
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