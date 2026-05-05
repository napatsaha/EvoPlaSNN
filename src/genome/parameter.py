from copy import copy

import numpy as np
from numpy.typing import ArrayLike

from common.base import Parameter


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
            length = value.size
            
            # # TODO: Validate value
            # if hasattr(value, "__len__"):
            #     length = len(value)
            # else:
            #     length = 1
        self._value = value
        self._length = length

    def _generate(self, size) -> ArrayLike:
        raise NotImplementedError()

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
    def __init__(self, value=None, length=1, low=0, high=1):
        self.low = low
        self.high= high
        # if value is None:
        #     value = self._generate(length)
        # else:
        #     # TODO: Validate value
        #     length = len(value)
        super().__init__(value, length)

    def _generate(self, size):
        return np.random.uniform(self.low, self.high, size)

    def __repr__(self):
        return f"RealParam({self.value}, low={self.low}, high={self.high})"

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
    def __init__(self, value=None, length=1, low=None, high=None):
        # if high is None:
        #     self.low = 0
        #     self.high = low
        # else:
        #     self.low = low
        #     self.high = high
        self.low = low
        self.high = high
        
        # if value is None:
        #     value = self._generate(length)
        # else:
        #     length = len(value)
        super().__init__(value, length)

    # @classmethod
    # def create(cls, low, high=None) -> 'DiscreteBounded':
    #     value = cls._generate(low, high)
    #     return cls(value, low, high)

    def _generate(self, size):
        return np.random.randint(self.low, self.high, size)

    # def sample(self):
    #     self._value = self._generate(self.low, self.high)
    #     return self.value

    def __repr__(self):
        return f"DiscreteParam({self.value}, low={self.low}, high={self.high})"
    
