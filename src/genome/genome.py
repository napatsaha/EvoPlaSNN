from typing import List, Literal

import numpy as np

from common.base import Genome, Parameter


class BaseGenome(Genome):
    """
    Base class to allow for genetic-related operations in evolutionary Solver.
    """
    def __init__(self, parameters = None, **kwargs):
        super().__init__()
        self._parameters = parameters

    def mutate(self, rate: float, dist: Literal["normal", "uniform"], **kwargs) -> 'Genome':
        """
        Create a modified copy of itself

        Args:
            rate (float): mutation rate
        """
        params = self.parameters.copy()
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        gene_to_mutate = np.random.randint(self.size, size=(int(rate*self.size), ))
        for gene_id in gene_to_mutate:
            if dist == "uniform":
                params[gene_id] = np.random.rand()
            elif dist == "normal":
                params[gene_id] = np.random.randn()
            else:
                raise ValueError(f"Distribution {dist} not supported")
        return self.__class__(params)

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

    def crossover(self, other: Genome, rate: float, return_genes_only: bool = False) -> Genome | List[Parameter]:
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
    
