"""
Cartesian Genetic Programming: graph program, and learning rule
"""

import numpy as np
from numpy.typing import ArrayLike
from common.base import LearningRule
from typing import List, Tuple, Literal, Callable


FUNCTION_LIST = [
    np.add,
    np.subtract,
    np.multiply,
    np.divide
]


class CGP_Graph:
    def __init__(self, 
                 n_inputs: int, n_outputs: int, n_rows: int, n_cols: int, *,
                 genome: list = None, function_list: List[Callable] = None,
                 arity: int = 2, prev_layer: int = None):
        self._ni = n_inputs
        self._no = n_outputs
        self._nr = n_rows
        self._nc = n_cols
        self._nn = self._nr * self._nc
        self._a = arity
        self._l = prev_layer if prev_layer is not None else self._nc

        # Allele length
        self._al = self._a + 1
        # Genome length
        self._lg = self._nn * self._al + self._no
        # Maximum nodes (inputs + internal)
        self._m = self._ni + self._nn

        # functions
        self.function_list = FUNCTION_LIST if function_list is None else function_list
        self._nf = len(self.function_list)

        # Internal arrays
        self._genome_internal = np.zeros((self._nr, self._nc, self._al), dtype=np.int16)
        self._genome_output = np.zeros((self._no), dtype=np.int16)

        # Init genome
        if genome is not None:
            self._fill_genome(genome)
            self._check_valid_genome()
        else:
            # TODO
            self._create_random_genome()

        # Find active nodes
        self._find_active_nodes()

        # Activation arrays
        # TODO: Figure out where and whether or not to setup num samples in output
        self._node_outputs = np.zeros((self._m + self._no), dtype=np.float64)

    def reset(self):
        self._node_outputs.fill(np.nan)

    def forward(self, inp: ArrayLike):
        self.reset()
        self._node_outputs[:self._ni] = inp

        for node in self._active_nodes:
            # Internal nodes
            # Get genome
            i, j = self.flat_to_coord(node - self._ni)
            gene = self._genome_internal[i, j, :]
            func = self.function_list[gene[-1]]
            inp_ix = gene[:self._a].flat
            inp = np.take(self._node_outputs, inp_ix)
            result = func(*inp)
            self._node_outputs[node] = result
        for i, out_node in enumerate(self._genome_output):
            # Output nodes
            result = self._node_outputs[out_node]
            self._node_outputs[i + self._m] = result

        return self._node_outputs[self._m:]

    def _create_random_genome(self):
        pass

    def _fill_genome(self, genome: list):
        # Check if genome is in tuple form or flat
        if len(genome) == (self._nn + self._no):
            # allele genome
            for o in range(self._no):
                self._genome_output[o] = genome[self._nn+o][0]
            for i in range(self._nn):
                allele = genome[i]
                self._genome_internal[i % self._nr, i // self._nr, :] = allele
        elif len(genome) == (self._lg):
            # flat genome
            self._genome_output[:] = genome[-self._no:]
            for i in range(self._nn):
                allele = genome[(self._al * i):(self._al * (i+1))]
                self._genome_internal[i % self._nr, i // self._nr, :] = allele
        else:
            raise ValueError("Length of genome does not match either (n_n + n_o) or (n_n*(a + 1) + n_o)")
        
    def _check_valid_genome(self):
        # TODO: More detailed error message by checking 2 clauses separately and using np.where
        # Check function
        fi = self._genome_internal[:, :, -1]
        assert np.all(0 <= fi) & np.all(fi < self._nf), "Function gene outside scope"
        # if not (0 <= self._genome_internal < self._nf):
            # idx = np.where(0 <= self._genome_internal < self._nf)
            # raise ValueError(f"Genome not outside function length {self._nf} at nodes: {idx}")
        # Check output
        assert np.all(0 <= self._genome_output) & np.all(self._genome_output < (self._ni + self._nn)), "Output genes invalid"
        # if not (self._nn <= self._genome_output < (self._nn + self._no)):
            # idx = np.where(self._nn <= self._genome_output < (self._nn + self._no)) + self._nn
            # raise ValueError(f"Invalid output gene at: {idx}")
        # Check connection genes
        for g in range(self._a):
            c = self._genome_internal[:, :, g]
            for j in range(self._nc):
                upper_bound = self._ni + j*self._nr
                if j >= self._l:
                    lower_bound = self._ni + (j - self._l)*self._nr
                    # assert np.all((self._ni + (j - self._l)*self._nr) <= c[:, j]) & np.all(c[:, j] < (self._ni + j*self._nr)), f"Connection gene invalid at column {j}"
                else:
                    lower_bound = 0
                    # assert np.all(0 <= c[:, j]) & np.all(c[:, j] < (self._ni + j*self._nr)), f"Connection gene invalid at column {j}"
                assert np.all(lower_bound <= c[:, j]) & np.all(c[:, j] < upper_bound), f"Connection gene invalid at column {j}"

    def _find_active_nodes(self):
        self._active_nodes = []
        self._active_nodes.extend(self._genome_output)
        for oi in self._active_nodes:
            i, j = self.flat_to_coord(oi - self._ni)
            gene = self._genome_internal[i, j, :]
            for node in gene[:-1]:
                if (node - self._ni) < 0:
                    continue
                if node not in self._active_nodes:
                    self._active_nodes.append(node)
        self._active_nodes = np.sort(np.array(self._active_nodes))
        
    def flat_to_coord(self, idx: int) -> tuple:
        i = idx % self._nr
        j = idx // self._nr
        return i, j

    def coord_to_flat(self, i: int, j: int) -> int:
        return j * self._nr + i


    def __equal__(self, other: 'CGP_Graph') -> bool:
        pass

    @property
    def n_inputs(self):
        return self._ni
    @property
    def n_nodes(self):
        return self._nn
    @property
    def n_outputs(self):
        return self._no
    @property
    def n_rows(self):
        return self._nr
    @property
    def n_cols(self):
        return self._nc
    @property
    def n_func(self):
        return self._nf
    @property
    def arity(self):
        return self._a
    @property
    def prev_layer(self):
        return self._l
    
    @property
    def genome(self):
        return np.r_[self._genome_internal.flatten(order="C"), self._genome_output]
    
    @property
    def active_nodes(self):
        return self._active_nodes



class CGP_Rule(LearningRule):
    def __init__(self):
        super().__init__()

    def update(self, always_return_tuple):
        return super().update(always_return_tuple)