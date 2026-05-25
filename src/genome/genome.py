from typing import List, Literal

import numpy as np
from numpy.typing import ArrayLike

from common.base import Genome, Parameter


class SimpleGenome(Genome):
    """
    Base class to allow for genetic-related operations in evolutionary Solver.
    """
    def __init__(self, parameters: ArrayLike = None, size: int = None, dist: Literal["normal", "uniform"] = "uniform", **kwargs):
        super().__init__()
        self.dist = dist
        if parameters is None:
            if size is not None:
                self._parameters = self._make_random_parameters(size)
            else:
                raise ValueError("Either 'parameters' or 'size' must be supplied.")
        else:
            self._parameters = parameters

    def _make_random_parameters(self, size: int) -> ArrayLike:
        if self.dist == "uniform":
            return np.random.random_sample(size=size)
        elif self.dist == "normal":
            return np.random.standard_normal(size=size)
        else:
            raise ValueError(f"Distribution {self.dist} not supported")

    def mutate(self, rate: float, **kwargs) -> 'Genome':
        """
        Create a modified copy of itself

        Args:
            rate (float): mutation rate
        """
        params = self.parameters.copy()
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        gene_to_mutate = np.random.randint(self.size, size=(int(rate*self.size), ))
        for gene_id in gene_to_mutate:
            params[gene_id] = self._make_random_parameters(size=1)
        return self.__class__(params)

    def crossover(self, other: Genome, rate: float) -> Genome:
        assert self.size == other.size, f"Both genome must have the same size. Got size={self.size} and size={other.size}"
        rate = np.clip(rate, 0, 1)
        flags = np.random.binomial(1, p=rate, size=self.size)
        new_params = np.where(flags, self.parameters, other.parameters)
        return self.__class__(parameters=new_params, dist=self.dist)

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

    @property
    def genome(self) -> Genome:
        return self

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

    def mutate(self, rate, return_genes_only: bool = False) -> Genome | List[Parameter]:
        new_genes = []
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        to_mutate_flag = np.random.binomial(1, rate, size=self.size)
        idx = 0
        for gene in self.genes:
            l = gene.length
            flags = to_mutate_flag[idx:(idx+l)]
            new_gene = gene.mutate(flags)
            new_genes.append(new_gene)
            idx += l

        if return_genes_only:
            return new_genes
        else:
            return self.__class__(new_genes)

    def crossover(self, other: 'CompositeGenome', rate: float, return_genes_only: bool = False) -> Genome | List[Parameter]:
        assert len(self.genes) == len(other.genes), "Both Composite Genomes must have the same number of genes. " + \
            f"Got {len(self.genes)} genes in first Genome and {len(other.genes)} genes in second Genome."
        assert self.size == other.size, f"Both genome must have the same size. Got size={self.size} and size={other.size}"
        new_genes = []
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        genome_flags = np.random.binomial(1, rate, size=self.size)
        idx = 0
        for gene, other_gene in zip(self.genes, other.genes):
            l = gene.length
            local_flags = genome_flags[idx:(idx+l)]
            new_gene = gene.crossover(other_gene, local_flags)
            new_genes.append(new_gene)
            idx += l

        if return_genes_only:
            return new_genes
        else:
            return self.__class__(new_genes)

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
    
