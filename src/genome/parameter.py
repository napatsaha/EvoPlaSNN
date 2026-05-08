from copy import copy
import sys
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from common.base import Parameter


PARAM_DICT = {
    "real": "RealParam",
    "discrete": "DiscreteParam"
}

def create_param(kind, **kwargs) -> Parameter:
    name = PARAM_DICT.get(kind)
    param_class = getattr(sys.modules[__name__], name)
    return param_class(**kwargs)


class BaseParameter(Parameter):
    length: int
    value: np.typing.ArrayLike

    def __init__(self, value, length: int = 1):
        super().__init__()
        if value is None:
            length = max(1, length)
            value = self._generate(length)
        else:
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
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

    def _generate(self, size) -> ArrayLike:
        raise NotImplementedError()
    
    def _validate(self, value: np.ndarray) -> np.ndarray:
        """
        Check for validity, return processed value and raise TypeError or ValueError

        Args:
            value (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: _description_
        """
        raise NotImplementedError(f"Each subclass must implement their own method of verifying value fits within distribution")

    def mutate(self, flags: bool | ArrayLike) -> 'BaseParameter':
        # if not isinstance(bool):
        #     assert len(flags) == self.length
        flags = np.asarray(flags, dtype=bool)
        assert flags.size == self.length, "Length of flags must be equal to length of param"
        # Make n number of changes according to how True there is in flags
        n_changes = sum(flags)
        replacement = self._generate(n_changes)
        # Identify location to swap
        idx_to_change = np.where(flags)
        # Insert changes into a copy of own value
        value = copy(self.value)
        np.put(value, idx_to_change, replacement)
        # Generate a new object based on new value
        new_param = self.__class__(value, **self.to_dict())
        return new_param

    def to_dict(self) -> dict:
        return {k: self.__dict__.get(k) for k in self.__dict__ if not k.startswith("_")}
    
    def copy(self) -> 'BaseParameter':
        value = copy(self._value)
        new_param = self.__class__(value, **self.to_dict())
        return new_param

    def __repr__(self):
        return f"Param({self._value})"
    
    @property
    def value(self):
        return self._value
    
    @property
    def length(self):
        return self._length


class RealParam(BaseParameter):
    _supported_dist = ("uniform", "normal")
    def __init__(self, value=None, length=1, *,
                 low=None, high=None, loc=None, scale=None,
                 dist: Literal["uniform", "normal"] = "uniform",
                 **kwargs):
        # Can be unbounded
        self.low = low #if low is not None else -np.inf
        self.high = high #if high is not None else np.inf
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
            if (low is None) or (high is None):
                raise ValueError("For 'uniform' distribution, both 'low' and 'high' boundaries must be provided. " + \
                                 f"Found low={low}, high={high}")
        elif dist == "normal":
            self._normal = True
            self.loc = loc if loc is not None else 0
            self.scale = scale if scale is not None else 1

        super().__init__(value, length)

    def _generate(self, size):
        if self._uniform:
            return np.random.uniform(self.low, self.high, size)
        elif self._normal:
            vals = np.random.normal(self.loc, self.scale, size)
            return np.clip(vals, self.low, self.high)
        else:
            return np.zeros(shape=size)
    
    def _validate(self, value):
        check_upper = (value < self.high) if self.high is not None else True
        check_lower = (value >= self.low) if self.low is not None else True
        check = check_lower & check_upper
        if not np.all(check):
            idx = np.where(~check)
            vals = value[idx]
            raise ValueError(f"All Values must be between [{self.low}, {self.high}). At index(s) {idx[0]}, got value(s) {vals}")
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


class DiscreteParam(BaseParameter):
    # DONE: Add n for number of possible values
    def __init__(self, value=None, length=1, low=None, high=None, n=None):
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

        super().__init__(value, length)

        # self._value = self._value.astype(np.int_)

    # @classmethod
    # def create(cls, low, high=None) -> 'DiscreteBounded':
    #     value = cls._generate(low, high)
    #     return cls(value, low, high)

    def _generate(self, size):
        return np.random.randint(self.low, self.high, size)

    def _validate(self, value: np.ndarray) -> np.ndarray:
        check = (value >= self.low) & (value < self.high)
        if not np.all(check):
            idx = np.where(~check)
            vals = value[idx]
            raise ValueError(f"All Values must be between [{self.low}, {self.high}). At index(s) {idx[0]}, got value(s) {vals}")
        return value.astype(np.int_)

    # def sample(self):
    #     self._value = self._generate(self.low, self.high)
    #     return self.value

    def __repr__(self):
        return f"DiscreteParam({self.value}, low={self.low}, high={self.high}, n={self.n})"
    
