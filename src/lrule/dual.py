from typing import List, Dict, Sequence
from copy import deepcopy

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

        
        self.internal_rule: EvolvableLearningRule = create_learning_rule(**deepcopy(internal_rule))
        self.external_rule: EvolvableLearningRule = create_learning_rule(**deepcopy(external_rule))

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

    def forward(self, inp):
        pass

    def update(self, synapse, reward, always_return_tuple):
        pass

    def crossover(self, other, rate = 0.5) -> 'DualLearningRule':
        offspring = super().crossover(other, rate)
        offspring._sync_genes()
        return offspring

    def mutate(self, rate = 1, scale = 0.1, method = "resample", **kwargs) -> 'DualLearningRule':
        offspring = super().mutate(rate, scale, method, **kwargs)
        offspring._sync_genes()
        return offspring