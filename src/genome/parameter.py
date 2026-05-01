from common.base import Parameter

import numpy as np


class BaseParameter(Parameter):
    def __init__(self, value):
        super().__init__()
        self._value = value

    def __repr__(self):
        return f"Param({self._value})"
    
    @property
    def value(self):
        return self._value


class UniformBoundedArray(BaseParameter):
    def __init__(self, value=None, size: int=None, low=0, high=1):
        self.size = size
        self.low = low
        self.high= high
        if value is None:
            value = self._generate(self.low, self.high, self.size)
        else:
            # TODO: Validate value
            pass
        super().__init__(value)

    @staticmethod
    def _generate(low, high, size):
        return np.random.uniform(low, high, size)


class UniformBounded(BaseParameter):
    def __init__(self, value=None, low=0, high=1):
        self.low = low
        self.high = high
        if value is None:
            value = self._generate(self.low, self.high)
        super().__init__(value)
        
    @staticmethod
    def _generate(low, high):
        return np.random.uniform(low, high)
    




class DiscreteBounded(BaseParameter):
    def __init__(self, value, low, high=None):
        if high is None:
            self.low = 0
            self.high = low
        else:
            self.low = low
            self.high = high
        super().__init__(value)

    @classmethod
    def create(cls, low, high=None) -> 'DiscreteBounded':
        value = cls._generate(low, high)
        return cls(value, low, high)

    @staticmethod
    def _generate(low, high):
        return np.random.randint(low, high)

    def sample(self):
        self._value = self._generate(self.low, self.high)
        return self.value

    def __repr__(self):
        return f"DiscreteBounded({self._value}, low={self.low}, high={self.high})"
    
