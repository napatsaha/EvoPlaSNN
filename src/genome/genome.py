from typing import List

import numpy as np

from common.base import Genome, Parameter


class BaseGenome(Genome):
    """
    Base class to allow for genetic-related operations in evolutionary Solver.
    """
    def __init__(self, parameters = None, **kwargs):
        super().__init__()
        self._parameters = parameters

    def mutate(self, rate: float) -> 'Genome':
        """
        Create a modified copy of itself

        Args:
            rate (float): mutation rate
        """
        pass

    @property
    def parameters(self) -> np.ndarray | List[Parameter]:
        """
        Returns a 1D genetic blueprint of the genome
        """
        return self._parameters

    @property
    def size(self) -> int:
        """
        Returns the number of parameters that exists in the genome
        """
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"Genome({self.parameters})"