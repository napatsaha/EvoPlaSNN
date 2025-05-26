from typing import Callable, List, Literal
from .base import LearningRule


import numpy as np



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
    "linear": lambda x: x
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
        
    def __repr__(self):
        return f"LinearLayer(input_size={self.input_size}, output_size={self.output_size}, activation_function={self.activation_function.__name__}, bias={self._bias})"


class ANN(LearningRule):
    """
    A LearningRule that is approximated by a fully-connected ANN.
    """
    def __init__(self, input_size: int, hidden_size: List | int = None, output_size: int = 1, 
                 parameters: List = None, bias: bool = True,
                 hidden_activation: str | Callable = None, output_activation: str | Callable = None,
                 weight_dist: Literal["uniform", "normal"] = "uniform"):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_sizes = self._solve_hidden_sizes(hidden_size)
        self.output_size = int(output_size)
        self.hidden_activation = _activation_functions.get(hidden_activation, relu)
        self.output_activation = _activation_functions.get(output_activation, linear)
        self.weight_dist = weight_dist
    
        self._create_layers(bias, parameters)

    def _solve_hidden_sizes(self, hidden_size: List | int | None) -> List[int | None]:
        """
        Solve the hidden sizes for the ANN.
        """
        if hidden_size is None or hidden_size == 0 or len(hidden_size) == 0:
            return []
        elif isinstance(hidden_size, int):
            return [hidden_size]
        elif isinstance(hidden_size, list):
            return hidden_size
        else:
            raise ValueError("hidden_size must be an int or a list of ints")
        
    def _create_layers(self, bias, parameters=None):
        self.layers: List[LinearLayer] = []
        self.layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.num_layers = len(self.layer_sizes)
        
        if parameters is not None:
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
        Compute the output of the ANN given the input x.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def update(self, w: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Apply the ANN Rule to an external set of weights.
        """
        dw = self.forward(x)
        if w is None:
            return dw
        else:
            return w + dw

    @property
    def parameters(self):
        """
        Return the evolvable parameters of the ANN.
        """
        params = []
        for layer in self.layers:
            params.append(layer.parameters)
        return np.concatenate(params)
    
    @property
    def weights(self):
        return [layer.weights for layer in self.layers]
    
    @property
    def biases(self):
        return [layer.bias for layer in self.layers]


    def __repr__(self):
        return f"ANN_Rule(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, output_size={self.output_size})"