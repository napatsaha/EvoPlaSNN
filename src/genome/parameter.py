from copy import copy
import sys
from typing import Any, Literal, Sequence, Dict
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from common.base import Parameter


PARAM_DICT = {
    "real": "RealParam",
    "log": "LogRealParam",
    "discrete": "DiscreteParam"
}

@dataclass
class GeneSpec:
    name: str
    kind: str
    length: int = 1
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: val for key, val in self.__dict__.items()}


def create_param(kind, **kwargs) -> Parameter:
    name = PARAM_DICT.get(kind)
    param_class = getattr(sys.modules[__name__], name)
    return param_class(**kwargs)


class BaseParameter(Parameter):
    length: int
    value: np.typing.ArrayLike
    name: str

    def __init__(self, value, length: int = 1, name: str = None, dtype: np.typing.DTypeLike = np.float32):
        super().__init__()
        if value is None:
            length = max(1, length)
            value = self._generate(length)
        else:
            if not isinstance(value, np.ndarray):
                value = np.asarray(value, dtype=dtype)
                # Force value to be at least 1D array (not 0D array) for consistent behaviour, e.g. len()
                if value.ndim == 0:
                    value = np.array([value])
            length = value.size
            
            # # TODO: Validate value
            try:
                value = self._validate(value)
            except Exception as e:
                raise e

        self._value = value
        self._length = length
        self._name = name if name is not None else ""

    def _generate(self, size) -> ArrayLike:
        raise NotImplementedError()
    
    def _perturb(self, location, scale: float) -> ArrayLike:
        delta = np.random.normal(0, scale=scale, size=self._length)
        value = np.where(location, self.value + delta, self.value)
        return value

    def _validate(self, value: np.ndarray) -> np.ndarray:
        """
        Check for validity, return processed value and raise TypeError / ValueError / Assertion Error

        Args:
            value (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: _description_
        """
        # NOTE: Cannot check for length, because during __init__, _validate is called before 'length' is set
        # assert self.length == value.size, f"Invalid length. New value has length {value.size}. This Parameter must have length {self.length}"
        return value

    def mutate(self, flags: bool | ArrayLike, method: Literal["resample", "perturb"] = "resample", scale: float = None) -> 'BaseParameter':
        # if not isinstance(bool):
        #     assert len(flags) == self.length
        if isinstance(flags, bool):
            flags = np.full(self.length, bool(flags), dtype=bool)
        elif isinstance(flags, (np.ndarray, Sequence)):
            assert flags.size == self.length, "Length of flags must be equal to length of param"
            flags = np.asarray(flags, dtype=bool)
        else:
            raise ValueError(f"Invalid flags argument: type={type(flags)}, value={flags}")
        # Make n number of changes according to how True there is in flags
        if method == "resample":
            n_changes = sum(flags)
            replacement = self._generate(n_changes)
            # Identify location to swap
            idx_to_change = np.where(flags)
            # Insert changes into a copy of own value
            value = copy(self.value)
            np.put(value, idx_to_change, replacement)
        elif method == "perturb":
            value = self._perturb(flags, scale)
        else:
            raise NotImplementedError(f"Mutation method {method} not yet supported. Only accept 'resample' or 'perturb'")
        # Generate a new object based on new value
        new_param = self.__class__(value, **self.to_dict())
        return new_param
    
    def crossover(self, other: 'Parameter', flags: bool | ArrayLike) -> 'BaseParameter':
        flags = np.asarray(flags, dtype=bool)
        assert flags.size == self.length, "Length of flags must be equal to length of param"
        assert other.length == self.length, "Cannot perform crossover between Parameters of different sizes"
        # Where flag==0, use this Parameter's value
        value_this = copy(self.value)
        value_other = copy(other.value)
        # Where flag==1, use the other Parameter's value
        value = np.where(flags, value_other, value_this)
        # Generate a new object based on new value
        new_param = self.__class__(value, **self.to_dict())
        return new_param

    def to_dict(self) -> dict:
        return {k: self.__dict__.get(k) for k in self.__dict__ if not k.startswith("_")}

    def get_value(self):
        return self.value
    
    def copy(self) -> 'BaseParameter':
        value = copy(self._value)
        new_param = self.__class__(value, **self.to_dict())
        return new_param

    def __repr__(self):
        return f"Param({self.value})"
    
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, new_value):
        if not isinstance(new_value, np.ndarray):
            new_value = np.asarray(new_value)
            # Force value to be at least 1D array (not 0D array) for consistent behaviour, e.g. len()
            if new_value.ndim == 0:
                new_value = np.array([new_value])
        assert new_value.size == self.length, f"Invalid length. New value has length {new_value.size}. This Parameter must have length {self.length}"
        self._value = self._validate(new_value)
    
    @property
    def length(self):
        return self._length

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value: str):
        assert isinstance(value, str)
        self._name = value


class RealParam(BaseParameter):
    _supported_dist = ("uniform", "normal")
    def __init__(self, value=None, length=1, name: str = None, *,
                 low=None, high=None, dist_params: dict = {},
                 dist: Literal["uniform", "normal"] = "uniform",
                 **kwargs):
        # Can be unbounded
        self.low = low #if low is not None else -np.inf
        self.high = high #if high is not None else np.inf
        self._bounded = (self.low is not None) or (self.high is not None)
        if (self.low is not None) and (self.high is not None): 
            if self.low >= self.high:
                raise ValueError(f"'low' must be less than 'high'. Got low={self.low}, high={self.high}")
        
        # Distribution
        self.dist = dist
        self._uniform = False
        self._normal = False
        if dist not in self._supported_dist:
            raise ValueError(f"'dist' must be one of {self._supported_dist}. Got {dist}")
        # Assign distribution for quick checks later on
        if dist == "uniform":
            self._uniform = True
            self.dist_params = {"low": dist_params.get("low", 0 if low is None else low),
                                "high": dist_params.get("high", 1 if high is None else high)}
            # if (low is None) or (high is None):
            #     raise ValueError("For 'uniform' distribution, both 'low' and 'high' boundaries must be provided. " + \
            #                      f"Found low={low}, high={high}")
        elif dist == "normal":
            self._normal = True
            self.dist_params = {"loc": dist_params.get("loc", kwargs.get("loc", 0)),
                                "scale": dist_params.get("scale", kwargs.get("scale", 1))}
            # self.loc = loc if loc is not None else 0
            # self.scale = scale if scale is not None else 1

        super().__init__(value, length, name, dtype=np.float32)

    def _generate(self, size):
        if self._uniform:
            return np.random.uniform(size=size, **self.dist_params)
        elif self._normal:
            vals = np.random.normal(size=size, **self.dist_params)
            if self._bounded:
                vals = np.clip(vals, self.low, self.high)
            return vals
        else:
            return np.zeros(shape=size)
    
    def _perturb(self, location, scale):
        value = super()._perturb(location, scale)
        value = np.clip(value, self.low, self.high)
        return value

    def _validate(self, value):
        value = super()._validate(value)
        if self._bounded:
            check_upper = (value <= self.high) if self.high is not None else True
            check_lower = (value >= self.low) if self.low is not None else True
            check = check_lower & check_upper
            if not np.all(check):
                idx = np.where(~check)
                vals = value[idx]
                raise ValueError(f"All Values must be between [{self.low}, {self.high}]. At index(s) {idx[0]}, got value(s) {vals}")
        return value

    def __repr__(self):
        kwargs = ', '.join(f"{k}={v}" for k, v in self.to_dict().items() if v is not None)
        return f"RealParam({self.value}, {kwargs})"

# class UniformBounded(BaseParameter):
#     def __init__(self, value=None, low=0, high=1):
#         self.low = low
#         self.high = high
#         if value is None:
#             value = self._generate(self.low, self.high)
#         super().__init__(value)
        
#     @staticmethod
#     def _generate(low, high):
#         return np.random.uniform(low, high)


class LogRealParam(RealParam):
    # TBA
    def __init__(self, value=None, length=1, name = None, *, 
                 base: float | str = None,
                 low=None, high=None, dist_params = None, dist = "normal", **kwargs):
        if dist_params is None:
            dist_params = {"loc":0, "scale":1}
        super().__init__(value, length, name, low=low, high=high, dist_params=dist_params, dist=dist, **kwargs)
        if base is None or (base == 'e'):
            self._base_e = True
            self._base_10 = False
            self.base = np.e
        else:
            assert isinstance(base, int | float | np.ndarray), f"Base must be a scalar number. Got type={type(base)}"
            self._base_e = np.isclose(base, np.e)
            self._base_10 = np.isclose(base, 10)
            self.base = np.e if self._base_e else 10 if self._base_10 else float(base)

    def _convert_value_out(self, value):
        """Return base^(value)"""
        if self._base_e:
            return np.exp(value)
        elif self._base_10:
            return 10**value
        else:
            return np.float_power(self.base, value)

    def _convert_value_in(self, value):
        """Return logarithm<base> of value"""
        if self._base_e:
            return np.log(value)
        elif self._base_10:
            return np.log10(value)
        else:
            return np.emath.logn(self.base, value)

    def get_value(self):
        return self._convert_value_out(self.value)

    def __repr__(self):
        kwargs = ', '.join(f"{k}={v}" for k, v in self.to_dict().items() if v is not None)
        return f"LogRealParam({self.value}, {kwargs})"


class DiscreteParam(BaseParameter):
    # DONE: Add n for number of possible values
    def __init__(self, value=None, length=1, name: str = None, *, 
                 low=None, high=None, n=None):
        # Make sure no value is missing assignment for some reason
        self.low = low
        self.high = high
        self.n = n
        # DONE: Deal with cases where low or high=None

        if n is not None:
            self.n = int(n)
            assert n > 0, "'n' must be at least 1"
            if (low is None) and (high is None):
                self.low = 0
                self.high = self.low + self.n
            elif (high is not None) and (low is None):
                self.high = int(high)
                self.low = self.high - self.n
            elif (low is not None) and (high is None):
                self.low = int(low)
                self.high = self.low + self.n
            else:
                assert high > low, "high must be greater than low"
                assert (high - low) == n, "Difference between 'high' and 'low' must be equal to 'n'"
                self.high = high
                self.low = low
        else:
            if (high is not None) and (low is not None):
                assert high > low, "high must be greater than low"
                self.low = int(low)
                self.high = int(high)
                self.n = self.high - self.low
            elif (low is None) and (high is not None):
                self.high = int(high)
                self.low = 0
                self.n = self.high - self.low
            else:
                raise AssertionError("Insufficient information. Either one of 'n' or 'high' or both of 'high' and 'low' must be passed through.")

        super().__init__(value, length, name, dtype=np.int_)

        # self._value = self._value.astype(np.int_)

    # @classmethod
    # def create(cls, low, high=None) -> 'DiscreteBounded':
    #     value = cls._generate(low, high)
    #     return cls(value, low, high)

    def _generate(self, size):
        return np.random.randint(self.low, self.high, size)

    def _perturb(self, location, scale):
        value = super()._perturb(location, scale)
        value = np.round(value, 0).astype(int)
        value = np.clip(value, self.low, self.high-1)
        return value

    def _validate(self, value: np.ndarray) -> np.ndarray:
        value = super()._validate(value)
        check = (value >= self.low) & (value < self.high)
        if not np.all(check):
            idx = np.where(~check)
            vals = value[idx]
            raise ValueError(f"All Values must be integers between [{self.low}, {self.high}). At index(s) {idx[0]}, got value(s) {vals}")
        return value.astype(np.int_)

    # def sample(self):
    #     self._value = self._generate(self.low, self.high)
    #     return self.value

    def __repr__(self):
        return f"DiscreteParam({self.value}, low={self.low}, high={self.high}, n={self.n})"
    
