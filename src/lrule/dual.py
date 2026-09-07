from typing import List, Dict, Sequence

from numpy.typing import ArrayLike

from common.base import LearningRule, Parameter
from genome.genome import EvolvableLearningRule
from common.utils import create_learning_rule


class DualLearningRule(LearningRule, EvolvableLearningRule):
    """
    A composite learning rule system which contains two learning rules in one genome.
    """
    def __init__(self, internal_rule: Dict, external_rule: Dict, *, 
                 parameters: ArrayLike = None, genes: List[Parameter] = None, 
                 genes_to_encode: Dict[str, Dict] = None, gene_order: Sequence[str] = None,
                 ):
        int_rule_type = internal_rule.pop("type")
        ext_rule_type = external_rule.pop("type")
        self.internal_rule: EvolvableLearningRule = create_learning_rule(int_rule_type, **internal_rule)
        self.external_rule: EvolvableLearningRule = create_learning_rule(ext_rule_type, **external_rule)

        # Update gene_order
        if gene_order is None:
            gene_order = ["external_rule", "internal_rule"]
        new_gene_order = []
        for g in gene_order:
            if g == "external_rule":
                new_gene_order.extend(self.external_rule.gene_order)
            elif g == "internal_rule":
                new_gene_order.extend(self.internal_rule.gene_order)
            else:
                new_gene_order.append(g)

        # Update gene_to_encode dict
        if genes_to_encode is None:
            genes_to_encode = {}
        genes_to_encode.update(self.internal_rule.genes_to_encode)
        genes_to_encode.update(self.external_rule.genes_to_encode)


        EvolvableLearningRule.__init__(self, parameters=parameters, genes=genes, genes_to_encode=genes_to_encode, gene_order=new_gene_order)

    def _build_gene_specs(self):
        specs = super()._build_gene_specs()
        specs.update(self.internal_rule.specs)
        specs.update(self.external_rule.specs)
        return specs

    def forward(self, inp):
        pass

    def update(self, synapse, reward, always_return_tuple):
        pass