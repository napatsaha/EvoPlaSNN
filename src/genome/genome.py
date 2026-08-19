from typing import Any, Dict, List, Literal, Sequence, Tuple, Optional
import warnings
import copy

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
        self._genes = genes
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
    def parameters(self, values: np.ndarray):
        assert len(values) == self.size
        i = 0
        for gene in self._genes:
            l = gene.length
            val = values[i:(i+l)]
            gene.value = val
            i += l
        self._parameters = np.r_[*[g.value for g in self._genes]]

    @property
    def genes(self) -> List[Parameter]:
        return self._genes

    @genes.setter
    def genes(self, value: List[Parameter]):
        self._genes = value
        self._parameters = np.r_[*[g.value for g in self._genes]]

    @property
    def size(self) -> int:
        return len(self.parameters)
    
    def __repr__(self):
        return "CompositeGenome(" + ', '.join([repr(g) for g in self.genes]) + ")"


class EvolvableLearningRule(Genome):
    genome: CompositeGenome
    _specs: Dict[str, Dict[str, Any]]
    default_gene_specs = {
        "learning_rate": dict(kind="real", length=1, low=0),
        "tau_syn": dict(kind="real", length=1, low=0, high=0.5, dist="uniform")
    }
    default_gene_order = ("learning_rate", "tau_syn")

    def __init__(self, *, parameters: ArrayLike = None, genes: List[Parameter] = None, 
                 genes_to_encode: List[Dict] = None, gene_order: Sequence[str] = None,
                 **kwargs):
        super().__init__()

        # Build specs based on default class GeneSpec
        # Specs control what encodings are possible by default in this subclass 
        self._specs = self._build_gene_specs()
        # Build gene order based on either 'gene_order', the order within genes_to_encode or default order
        # Gene order controls how values in genome or list of genes should be read or written
        self._gene_order = self._build_gene_order(gene_order, genes_to_encode)
        # Build gene params from 'genes_to_encode' or, if absent, class GeneSpec
        # gene_params contain kwargs for each gene Parameter 
        genes_to_encode = self._normalise_gene_params(genes_to_encode)
        self._gene_params = self._build_gene_params(genes_to_encode)

        # Build genes and encoded values from input data depending on what is passed in
        self._genes: List[Parameter] = []
        # self._values: Dict[str, Any] = {}
        # CASE 1 -> Reconstruct from flat genome
        if parameters is not None:
            self._genes = self._genes_from_parameters(parameters)
        # CASE 2 -> Build from existing genes
        elif genes is not None:
            self._genes = self._genes_from_objects(genes)
        # CASE 3 -> Sample from gene spec
        # elif genes_to_encode is not None:
        #     self._genes, self._values = self._genes_from_templates(genes_to_encode)
        # CASE 4 -> Fallback, randomising from default specs
        else:
            self._genes = self._random_genes()

        # self._gene_lookup = {spec.name: gene for spec, gene in zip(self._specs, self._genes)}

        # Build genome
        self.genome = CompositeGenome(genes=self._genes)

        # Extract values from genes and apply values to self attributes
        self._values = self._extract_values_from_genes()
        self._apply_gene_values()

    def _build_gene_specs(self) -> Dict[str, Dict[str, Any]]:
        specs = {}
        specs.update(self.default_gene_specs.items())
        # specs.extend(self.rule_specific_gene_specs())
        return specs

    def _build_gene_order(self, gene_order: Sequence[str] = None, genes_to_encode: List[Dict] = None) -> List[str]:
        if gene_order is not None:
            for gene in gene_order:
                if gene not in self._specs:
                    raise RuntimeError(f"Gene: {gene} not supported by this class")
            return tuple(gene_order)
        elif genes_to_encode is not None:
            order = []
            if not isinstance(genes_to_encode, List):
                warnings.warn(f"The passed-in 'genes_to_encode' is not a List, so 'gene_order' cannot be read from it.")
                return tuple(self.default_gene_order)
            for item in genes_to_encode:
                name = item.get("name")
                if name is not None:
                    order.append(name)
                else:
                    raise ValueError(f"Name field must be defined for each entry in genes_to_encode")
            return tuple(order)
        else:
            return tuple(self.default_gene_order)

    def _normalise_gene_params(self, genes_to_encode: List[Dict] | Dict[str, Dict] = None) -> Dict[str, Dict[str, Any]] | None:
        """
        Convert config of gene_params to easier format of Dict[name: Dict[params]]. 
        """
        if genes_to_encode is None:
            return None
        if isinstance(genes_to_encode, Dict):
            # genes_to_encode is already in Dict format. No need to convert
            return genes_to_encode
        params = {}
        for i, item in enumerate(genes_to_encode):
            assert isinstance(item, Dict), "Each entry within 'genes_to_encode' must be a dictionary"
            assert "name" in item, f"Entry #{i} of genes_to_encode does not have a 'name' field"
            name = item.pop("name")
            params[name] = dict(item)
        return params
    
    def _build_gene_params(self, genes_to_encode: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        params = {}
        for gene_name in self._gene_order:
            default_params = self._specs.get(gene_name).copy()
            if "name" in default_params:
                default_params.pop("name") # Redundant information
            # Override existing params with user-input params
            if (genes_to_encode is not None) and (gene_name in genes_to_encode):
                new_params = genes_to_encode.get(gene_name)
                default_params.update(new_params)
            params[gene_name] = default_params
        return params

    def rule_specific_gene_specs(self):
        return []

    def _genes_from_parameters(self, parameters: ArrayLike) -> List[Parameter]:
        genes = []
        # values = {}
        i = 0

        for gene_name in self._gene_order:
            gene_params = self._gene_params[gene_name].copy()
            l = gene_params.get("length")
            value = parameters[i:(i+l)]
            kind = gene_params.pop("kind")
            # gene_params.pop("default")
            # gene_params.pop("name")
            # TODO: Validate value with gene_params
            gene = param.create_param(kind=kind, value=value, **gene_params)
            gene.name = gene_name
            # values[gene_name] = value
            i += l
            genes.append(gene)

        return genes#, values

    def _genes_from_objects(self, new_genes: List[Parameter]) -> List[Parameter]:
        genes = []
        # values = {}
        i = 0
        for gene_name in self._gene_order:
            gene_params = self._gene_params[gene_name]
            new_gene = new_genes[i]
            # value = self._simplify_value(new_gene.value, gene_spec.length)
            # value = new_gene.value
            # values[gene_name] = value
            # TODO: Validate gene with gene_params
            # Do something with gene_spec and new_gene
            new_gene.name = gene_name
            genes.append(new_gene)
            i += 1

        return genes#, values

    # def _genes_from_templates(self, genes_to_encode: List[Dict]) -> Tuple[List[Parameter], Dict[str, Any]]:
    #     return [], {}

    def _random_genes(self, ) -> List[Parameter]:
        genes = []
        # values = {}
        i = 0
        for gene_name in self._gene_order:
            gene_params = self._gene_params[gene_name].copy()
            # kwargs = {key: val for key, val in gene_spec.__dict__.items() if key != "kind"}
            kind = gene_params.pop("kind")
            # value = gene_params.pop("default")
            # gene_params.pop("name")
            gene = param.create_param(kind=kind, **gene_params)
            # value = gene.value
            # values[gene_name] = value
            gene.name = gene_name
            i += 1
            genes.append(gene)
        return genes#, values

    def _simplify_value(self, value, length):
        return value[0] if length == 1 and isinstance(value, np.ndarray | Sequence) and len(value) == 1 else value

    def _extract_values_from_genes(self) -> Dict[str, ArrayLike]:
        """
        Extract and update values from genome according to gene order.

        Returns Dictionary[gene_name: gene_value]
        """
        values = {}
        for gene_name, gene in zip(self.gene_order, self.genes):
            values[gene_name] = gene.value
        return values

    def _apply_gene_values(self):
        """
        Perform class-specific operations based on gene values
        """
        if self.encode_learning_rate:
            self.learning_rate = self.values.get("learning_rate")

    # def _apply_specific_gene_values(self):
    #     pass

    def mutate(self, rate: float = 1.0, scale: float = 0.1, method: Literal["resample", "perturb"] = "resample", **kwargs) -> 'EvolvableLearningRule':
        dup = self.copy()
        genes = dup.genome.mutate(rate, scale, method, return_genes_only=True, **kwargs)
        dup.genes = genes
        # Update internal values from new gene to ensure they are not retained from previous copy
        dup._values = dup._extract_values_from_genes()
        dup._apply_gene_values()
        return dup

    def crossover(self, other: 'EvolvableLearningRule', rate: float = 0.5) -> 'EvolvableLearningRule':
        child = self.copy()
        genes = child.genome.crossover(other.genome, rate, return_genes_only=True)
        child.genes = genes
        # Update internal values from new gene to ensure they are not retained from previous copy
        child._values = child._extract_values_from_genes()
        child._apply_gene_values()
        return child

    def copy(self) -> 'EvolvableLearningRule':
        return copy.deepcopy(self)
    
    @property
    def parameters(self) -> np.ndarray:
        """
        Flattened array of concatenated genome values
        """
        return self.genome.parameters
    @parameters.setter
    def parameters(self, values):
        self.genome.parameters = values

    @property
    def gene_order(self) -> List[str]:
        """
        Gene order controls how values in genome or list of genes should be read or written
        """
        return self._gene_order

    @property
    def specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Specs control what encodings are possible by default in this subclass.  
        (see `genes_to_encode` for actual specs after user update)
        """
        return self._specs

    @property
    def genes_to_encode(self) -> Dict[str, Dict[str, Any]]:
        """
        `genes_to_encode` contain kwargs for each gene Parameter
        """
        return self._gene_params

    @property
    def genes(self) -> List[Parameter]:
        """
        This is a list of Parameters of encoded values for evolution, in the order specified by `gene_order`.
        """
        return self._genes
    @genes.setter
    def genes(self, new_genes: List[Parameter]):
        assert len(new_genes) == len(self.gene_order), f"New genes must have length {len(self.gene_order)}, but got length {len(new_genes)} instead"
        self._genes = new_genes
        if hasattr(self.genome, "genes"):
            self.genome.genes = new_genes

    @property
    def values(self) -> Dict[str, ArrayLike]:
        """
        Dictionary of gene value by name of each encoded gene, for easy access.
        """
        return self._values

    @property
    def encode_learning_rate(self) -> bool:
        return "learning_rate" in self._gene_order
    @property
    def encode_tau_syn(self) -> bool:
        return "tau_syn" in self._gene_order
