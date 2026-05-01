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
    def parameters(self) -> np.ndarray:
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
    

class CompositeGenome(Genome):
    """
    A Genome which consists of a collection of genes of varying sizes. Each gene is a parameter.
    The entire genome is constructed from concatenating each gene's parameters together.
    """
    genes: List[Parameter]

    def __init__(self, genes: List[Parameter], **kwargs):
        super().__init__(**kwargs)
        self.genes = genes
        param = [g.value for g in self.genes]
        self._parameters = np.r_[*param]

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError(f"Parameter setter for {self.__class__.__name__} not yet implemented")

    @property
    def size(self) -> int:
        return len(self.parameters)
    
    def __repr__(self):
        return "CompositeGenome(" + ', '.join([repr(g) for g in self.genes]) + ")"
    
