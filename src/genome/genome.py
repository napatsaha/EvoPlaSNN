from typing import Any, Dict, List, Literal, Sequence, Tuple, Optional

from genome import parameter as param
from genome.parameter import GeneSpec
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

    def mutate(self, rate: float, scale: float, method: Literal["resample", "perturb"], **kwargs) -> 'Genome':
        """
        Create a modified copy of itself

        Args:
            rate (float): mutation rate
        """
        params = self.parameters.copy()
        if method == "resample":
            rate = np.clip(rate, 0, 1, dtype=np.float32)
            gene_to_mutate = np.random.randint(self.size, size=(int(rate*self.size), ))
            for gene_id in gene_to_mutate:
                params[gene_id] = self._make_random_parameters(size=1)
        elif method == "perturb":
            delta = np.random.normal(0, scale=scale, size=self.size)
            params = params + delta
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

    def mutate(self, rate: float, scale: float, method: Literal["resample", "perturb"], *, 
               return_genes_only: bool = False) -> Genome | List[Parameter]:
        new_genes = []
        rate = np.clip(rate, 0, 1, dtype=np.float32)
        to_mutate_flag = np.random.binomial(1, rate, size=self.size)
        idx = 0
        for gene in self.genes:
            l = gene.length
            flags = to_mutate_flag[idx:(idx+l)]
            new_gene = gene.mutate(flags=flags, method=method, scale=scale)
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


class EvolvableLearningRule(Genome):
    genome: Genome
    _specs: Dict[str, GeneSpec]
    universal_gene_specs = [
        GeneSpec("learning_rate", kind="real", length=1, default=1.0),
    ]
    gene_order = ("learning_rate", )

    def __init__(self, *, parameters: ArrayLike = None, genes: List[Parameter] = None, genes_to_encode: List[Dict] = None,
                 **kwargs):
        super().__init__()

        # Read gene specs from universal and rule-specific genes
        self._specs = self._get_gene_specs()
        self._gene_params, self.gene_order = self._build_gene_params(self._specs, genes_to_encode)
        # self._specs = self._update_gene_specs(self._specs, self._gene_params)

        # TODO: Add system for setting which gene should be enabled
        # self.encode_learning_rate = encode_learning_rate
        # self.encode_tau_syn = encode_tau_syn

        # self.encodings = [self.encode_learning_rate, self.encode_tau_syn]

        # Build genes from input data
        self._genes: List[Parameter] = []
        self._values: Dict[str, Any] = {}
        # CASE 1 -> Reconstruct from flat genome
        if parameters is not None:
            self._genes, self._values = self._genes_from_parameters(parameters)
        # CASE 2 -> Build from existing genes
        elif genes is not None:
            self._genes, self._values = self._genes_from_objects(genes)
        # CASE 3 -> Sample from gene spec
        # elif genes_to_encode is not None:
        #     self._genes, self._values = self._genes_from_templates(genes_to_encode)
        # CASE 4 -> Fallback, randomising from default specs
        else:
            self._genes, self._values = self._random_genes()

        # self._gene_lookup = {spec.name: gene for spec, gene in zip(self._specs, self._genes)}

        # Build genome
        self.genome = CompositeGenome(genes=self._genes)

        # Apply genes
        self._apply_gene_values()

    def add_encodings(self, genes_to_encode: List):
        pass

    def _get_gene_specs(self) -> Dict[str, GeneSpec]:
        specs = {}
        specs.update({spec.name: spec for spec in self.universal_gene_specs})
        # specs.extend(self.rule_specific_gene_specs())
        return specs

    def _build_gene_params(self, specs: Dict[str, GeneSpec], genes_to_encode: Optional[List[Dict]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Convert config of gene_params to easier format of Dict[name: Dict[params]]. 
        A new order will be read from the list order of `genes_to_encode` if its name field is present,
        otherwise the default class attribute `gene_order` will be used.
        """
        params = {}
        order = []
        if genes_to_encode is None:
            order = self.gene_order
            params = {name: specs[name].to_dict() for name in order}
            return params, order

        else:

            for i, item in enumerate(genes_to_encode):
                new_params = dict(item)
                name = new_params.pop("name")
                if name is None:
                    raise ValueError(f"Name must be defined in entry #{i} of genes_to_encode")
                existing_params = specs.get(name).to_dict()
                new_params.update(existing_params)
                params[name] = new_params
                order.append(name)
        return params, order

    def rule_specific_gene_specs(self):
        return []

    def _genes_from_parameters(self, parameters: ArrayLike) -> Tuple[List[Parameter], Dict[str, Any]]:
        genes = []
        values = {}
        i = 0

        for gene_name in self.gene_order:
            gene_params = self._gene_params[gene_name]
            l = gene_params.get("length")
            value = parameters[i:(i+l)]
            kind = gene_params.pop("kind")
            gene_params.pop("default")
            gene_params.pop("name")
            # value = self._simplify_value(value, length=l)
            # kwargs = {key: val for key, val in gene_spec.__dict__.items() if key != "kind"}
            gene = param.create_param(kind=kind, value=value, **gene_params)

            values[gene_name] = value
            i += l
            genes.append(gene)

        return genes, values

    def _genes_from_objects(self, new_genes: List[Parameter]) -> Tuple[List[Parameter], Dict[str, Any]]:
        genes = []
        values = {}
        i = 0
        for gene_name in self.gene_order:
            # gene_params = self._gene_params[gene_name]
            new_gene = new_genes[i]
            # value = self._simplify_value(new_gene.value, gene_spec.length)
            value = new_gene.value
            values[gene_name] = value
            # TODO: Validate gene
            # Do something with gene_spec and new_gene
            new_gene.name = gene_name
            genes.append(new_gene)
            i += 1

        return genes, values

    # def _genes_from_templates(self, genes_to_encode: List[Dict]) -> Tuple[List[Parameter], Dict[str, Any]]:
    #     return [], {}

    def _random_genes(self, ) -> Tuple[List[Parameter], Dict[str, Any]]:
        genes = []
        values = {}
        i = 0
        for gene_name in self.gene_order:
            gene_params = self._gene_params[gene_name]
            # kwargs = {key: val for key, val in gene_spec.__dict__.items() if key != "kind"}
            kind = gene_params.pop("kind")
            value = gene_params.pop("default")
            gene_params.pop("name")
            gene = param.create_param(kind=kind, value=value, **gene_params)
            value = gene.value
            values[gene_name] = value
            i += 1
            genes.append(gene)
        return genes, values

    def _simplify_value(self, value, length):
        return value[0] if length == 1 and isinstance(value, np.ndarray | Sequence) and len(value) == 1 else value

    def _apply_gene_values(self):
        # TODO: Extract universal gene values

        # Apply gene values to rule-specific scenarios
        self._apply_specific_gene_values()

    def _apply_specific_gene_values(self):
        pass

    def mutate(self, rate, scale, method, **kwargs):
        raise NotImplementedError()

    def crossover(self, other, rate):
        raise NotImplementedError()
    
