from typing import Any, List, Dict, Sequence, Tuple
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike

from common.base import LearningRule, Parameter, Genome, EvolvableRuleProtocol
from genome.genome import EvolvableLearningRule
from common.utils import create_learning_rule


class DualLearningRule(LearningRule, EvolvableLearningRule):
    """
    A composite learning rule system which contains two learning rules in one genome.
    """
    def __init__(self, internal_rule: EvolvableLearningRule, external_rule: EvolvableLearningRule, *, 
                 parameters: ArrayLike = None, genes: List[Parameter] = None, 
                 genes_to_encode: Dict[str, Dict] = None, gene_order: Sequence[str] = None,
                 ):

        
        self.internal_rule = internal_rule
        self.external_rule = external_rule

        # Deal with possibility of duplicate gene names
        # -> The only important part to identify is in 'gene_order' of each rule
        ext_genes = self.external_rule.gene_order
        int_genes = self.internal_rule.gene_order

        # Need to change the gene names of 3 things: gene_order, genes_to_encode, and specs
        new_ext_gene_order = [str(g) + "_ext" if g in int_genes else g for g in ext_genes]
        new_ext_gene_params = {str(g)+"_ext" if g in int_genes else g: v for g,v in self.external_rule.genes_to_encode.items()}
        self._ext_specs = {str(g)+"_ext" if g in int_genes else g: v for g,v in self.external_rule.specs.items()}

        new_int_gene_order = [str(g) + "_int" if g in ext_genes else g for g in int_genes]
        new_int_gene_params = {str(g)+"_int" if g in ext_genes else g: v for g,v in self.internal_rule.genes_to_encode.items()}
        self._int_specs = {str(g)+"_int" if g in ext_genes else g: v for g,v in self.internal_rule.specs.items()}

        # Update gene_order
        if gene_order is None:
            gene_order = ["external_rule", "internal_rule"]
        new_gene_order = []
        self._gene_origin_flag = []
        for gene in gene_order:
            if gene == "external_rule":
                new_gene_order.extend(new_ext_gene_order)
                self._gene_origin_flag.extend(["external" for _ in range(len(new_ext_gene_order))])
            elif gene == "internal_rule":
                new_gene_order.extend(new_int_gene_order)
                self._gene_origin_flag.extend(["internal" for _ in range(len(new_int_gene_order))])
            else:
                new_gene_order.append(gene)
                self._gene_origin_flag.append("shared")

        # Update gene_to_encode dict
        if genes_to_encode is None:
            genes_to_encode = {}
        genes_to_encode.update(new_int_gene_params)
        genes_to_encode.update(new_ext_gene_params)


        EvolvableLearningRule.__init__(self, parameters=parameters, genes=genes, genes_to_encode=genes_to_encode, gene_order=new_gene_order)

        # Updates internal/external rules' genes with the one from this instance
        self._sync_genes()

    def _sync_genes(self):
        self.internal_rule.genes = [g for g, flag in zip(self.genes, self._gene_origin_flag) if flag == "internal"]
        self.external_rule.genes = [g for g, flag in zip(self.genes, self._gene_origin_flag) if flag == "external"]

    def _build_gene_specs(self):
        specs = super()._build_gene_specs()
        specs.update(self._int_specs)
        specs.update(self._ext_specs)
        return specs

    @classmethod
    def from_params(cls, internal_rule: Dict, external_rule: Dict, **kwargs):
        internal_rule: EvolvableLearningRule = create_learning_rule(**deepcopy(internal_rule))
        external_rule: EvolvableLearningRule = create_learning_rule(**deepcopy(external_rule))

        return cls(internal_rule, external_rule, **kwargs)

    def forward(self, inp):
        pass

    def update(self, synapse, reward, always_return_tuple):
        pass

    def crossover(self, other, rate = 0.5) -> 'DualLearningRule':
        offspring: 'DualLearningRule' = super().crossover(other, rate)
        offspring._sync_genes()
        return offspring

    def mutate(self, rate = 1, scale = 0.1, method = "resample", **kwargs) -> 'DualLearningRule':
        offspring: 'DualLearningRule' = super().mutate(rate, scale, method, **kwargs)
        offspring._sync_genes()
        return offspring

    def __repr__(self):
        return f"DualLearningRule(\n\tinternal_rule={self.internal_rule}, \n\texternal_rule={self.external_rule}, " + \
            f"gene_order={self.gene_order}\n)"


class LayeredRule(LearningRule, EvolvableRuleProtocol):
    """
    Container with separate rule for each synapse layer.
    """
    def __init__(self, rules: List[LearningRule | Genome], layers: List[int] = None):
        super().__init__()
        for i, r in enumerate(rules):
            assert isinstance(r, LearningRule), f"Item no {i} must be a learning rule. Got type={type(r)}"
        self._rules = rules

        if layers is None:
            self._layers = [*range(len(self._rules))]
        else:
            assert len(layers) == len(self._rules), f"Length of layers ({len(layers)}) must be the same as number of rules ({len(self._rules)})"
            self._layers = layers

    def get_rule_from_layer(self, layer_id: int) -> LearningRule | None:
        if layer_id not in self._layers:
            return None
        else:
            idx = self._layers.index(int(layer_id))
            return self._rules[idx]

    @classmethod
    def from_params(cls, rules: List[dict], layers: List[int] = None, 
                    parameters: ArrayLike = None, genes: List[Parameter] = None, 
                    **kwargs):
        """
        Construct a LayeredRule from a list of lrule param dictionaries, instead of learning rule objects

        Args:
            rules (List[dict]): List of learning rule dictionaries
            layers (List[int], optional): SynapseLayer number each rule will apply to. If empty, will generate layer id sequentially. Defaults to None.
            parameters (ArrayLike, optional): Genome array to construct. Defaults to None.
            genes (List[Parameter], optional): List of genes to construct each constituent rule. Defaults to None.

        Returns:
            LayeredRule: a LayeredRule object
        """
        rule_list = []
        i = 0
        for r_params in rules:
            rule = create_learning_rule(**r_params)
            if parameters is not None:
                # Since it is difficult to estimate genome size before creating the rule, to use for slicing 'parameters',
                #  temp solution is to replace the parameters of each rule after creation
                if not isinstance(rule, Genome):
                    # Skip parsing parameters if rule is a genome-less learning rule
                    continue
                l = rule.size
                p = parameters[i:(i+l)]
                rule.parameters = p
                i += l
            elif genes is not None:
                if not isinstance(rule, EvolvableRuleProtocol):
                    # Skip parsing parameters if genes is a gene-less learning rule
                    continue
                l = len(rule.genes)
                g = genes[i:(i+l)]
                rule.genes = g
                i += l
                # raise NotImplementedError("Setting 'genes' during init is not yet supported in LayeredRule")
            rule_list.append(rule)
        # rule_list = [create_learning_rule(**r_params) for r_params in rules]
        return cls(rules=rule_list, layers=layers, **kwargs)

    def forward(self):
        raise TypeError("LayeredRule does not support 'forward' method. Please call 'forward' of constituent rule individually.")

    def update(self):
        raise TypeError("LayeredRule does not support 'update' method. Please call 'update' of constituent rule individually.")

    def mutate(self, rate: float, scale: float, method: str, **kwargs):
        new_rules = []
        for rule in self.rules:
            new_rule = rule.mutate(rate, scale, method, **kwargs)
            new_rules.append(new_rule)
        return LayeredRule(rules=new_rules, layers=self.layers)
        # raise NotImplementedError()

    def crossover(self, other: 'LayeredRule', rate: float):
        assert len(self.rules) == len(other.rules), "Both LayeredRules must contain the same number of rules. " + f"Got {len(self.rules)} rules in first, and {len(other.rules)} rules in other"
        assert self.size == other.size, "Both LayeredRules must have the same genome size. " + f"Got size={self.size} for first, and size={other.size} for other"
        # raise NotImplementedError()
        new_rules = []
        for rule1, rule2 in zip(self.rules, other.rules):
            new_offspring = rule1.crossover(rule2, rate=rate)
            new_rules.append(new_offspring)
        return LayeredRule(rules=new_rules, layers=self.layers)

    def copy(self) -> 'LayeredRule':
        return deepcopy(self)

    def get_bounds(self) -> Tuple[List, List]:
        lower, upper = [], []
        for rule in self.rules:
            low, upp = rule.get_bounds()
            lower.extend(low)
            upper.extend(upp)
        return lower, upper

    def contains_gene(self, gene_name: str) -> bool:
        return gene_name in self.gene_order

    def _rename_gene(self, layer_i, gene_name):
        return f"{gene_name}_layer{layer_i}"

    @property
    def rules(self) -> List[EvolvableRuleProtocol]:
        return self._rules

    @property
    def layers(self) -> List[int]:
        return self._layers

    @property
    def parameters(self) -> np.ndarray:
        return np.array([p for r in self.rules for p in r.parameters])
    @parameters.setter
    def parameters(self, value: np.ndarray):
        assert len(value) == self.size, f"Parameter length must be {self.size}. Got length={len(value)}"
        i=0
        for rule in self.rules:
            l = rule.size
            p = value[i:(i+l)]
            rule.parameters = p
            i += l

    @property
    def genes(self) -> List[Genome]:
        return np.array([g for r in self.rules for g in r.genes])

    @property
    def gene_order(self) -> List[str]:
        return [self._rename_gene(i, g) for i, r in zip(self.layers, self.rules) for g in r.gene_order]

    @property
    def values(self) -> Dict[str, ArrayLike]:
        return {self._rename_gene(i, g): v for i, r in zip(self.layers, self.rules) for g, v in r.values.items()}

    @property
    def genes_to_encode(self) -> Dict[str, Dict[str, Any]]:
        return {self._rename_gene(i, g): p for i, r in zip(self.layers, self.rules) for g, p in r.genes_to_encode.items()}

    @property
    def size(self) -> int:
        return sum([r.size for r in self.rules])

    def __repr__(self) -> str:
        return "LayeredRule(rules=[\n\t" + \
            ",\n\t".join([repr(r).replace("\n", "\n\t") for r in self.rules]) + "\n\t]," + \
            f"\n\tlayers={self.layers}" + \
            "\n)"

