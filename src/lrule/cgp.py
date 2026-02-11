"""
Cartesian Genetic Programming: graph program, and learning rule
"""

from lrule.utils import tile_array
import numpy as np
from numpy.typing import ArrayLike
from common.base import LearningRule, SynapseLayerProtocol
from lrule.base import BaseLearningRule
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
                 arity: int = 2, prev_layer: int = None, seed: int = None):
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
            self._create_random_genome(seed=seed)

        # Find active nodes
        self._find_active_nodes()

        # Activation arrays
        # Starts with empty dimension in axis=1 to allow broadcasting to fit input size if needed
        self._node_outputs = np.zeros((self._m + self._no, 1), dtype=np.float64)

    def reset(self, ns_in: int = None):
        self._node_outputs.fill(np.nan)
        # Perform output array reshaping to match input sample size
        if ns_in is not None: 
            ns_curr = self._node_outputs.shape[1]
            in_size = self._node_outputs.shape[0]
            # Check if current sample size is compatible with new sample size
            if ns_in == ns_curr:
                # If both equal, nothing needed
                pass
            elif ns_in == 1:
                # If new size is one, just slice current one
                self._node_outputs = self._node_outputs[:, :1].copy()
            # elif ns_in > ns_curr and ns_curr == 1:
            #     self._node_outputs = np.broadcast_to(self._node_outputs, (in_size, ns_in)).copy()
            #     # self._node_outputs.setflags(write=True)
            else:
                # Broadcasting only works if either one dimension is 1 or they are equal (doesn't work by divisibility)
                # Hence, every other condition requires stripping previous array to (x, 1) before broadcasting
                self._node_outputs = np.broadcast_to(self._node_outputs[:, :1], (in_size, ns_in)).copy()
                # a copy is returned to allow flags (OWNDATA, WRITEABLE) to be True
            # else:
            #     raise ValueError(f"Broadcasting not possible between input array: (ni, {ns_in}) and current output array: {self._node_outputs.shape}")


    def forward(self, inp: ArrayLike, squeeze: bool = False):
        # inp = np.squeeze(inp) # Remove empty dimension
        inp = inp.reshape(self._ni, -1) # Remove superfluous dimensions
        ns = 1 if inp.ndim == 1 else inp.shape[-1] # assume sample is in last dimension after squeezing
        self.reset(ns_in=ns)
        self._node_outputs[:self._ni, :] = inp

        for node in self._active_nodes:
            # Internal nodes
            # Get genome
            i, j = self.flat_to_coord(node - self._ni)
            gene = self._genome_internal[i, j, :]
            func = self.function_list[gene[-1]]
            inp_ix = gene[:self._a].flat
            inp = np.take(self._node_outputs, inp_ix, axis=0)
            result = func(*inp)
            self._node_outputs[node, :] = result
        for i, out_node in enumerate(self._genome_output):
            # Output nodes
            result = np.take(self._node_outputs, out_node, axis=0)
            self._node_outputs[i + self._m, :] = result

        if not squeeze:
            return self._node_outputs[self._m:, :]
        else:
            return np.squeeze(self._node_outputs[self._m:, :])

    def _create_random_genome(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        # Randomise connection genes
        for j in range(self._nc):
            upper_bound = self._ni + j*self._nr
            lower_bound = 0 if j < self._l else self._ni + (j - self._l)*self._nr
            req_shape = self._genome_internal[:, j, :self._a].shape
            self._genome_internal[:, j, :self._a] = np.random.randint(lower_bound, upper_bound, size=req_shape)
        # Randomise function genes
        req_shape = self._genome_internal[:, :, self._a].shape
        self._genome_internal[:, :, self._a] = np.random.randint(0, self._nf, size=req_shape)
        # Randomise output genes
        self._genome_output[:] = np.random.randint(0, self._m, size=self._genome_output.shape)
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
    def size(self):
        "Complete genome size : (n_outputs + n_nodes * (arity + 1))"
        return self._lg

    @property
    def genome(self):
        return np.r_[self._genome_internal.swapaxes(0, 1).flatten(order="C"), self._genome_output]
    
    @genome.setter
    def genome(self, value):
        self._genome_internal.fill(0)
        self._genome_output.fill(0)
        self._fill_genome(value)
        self._check_valid_genome()

    @property
    def active_nodes(self):
        return self._active_nodes



class CGP_Rule(BaseLearningRule):
    """
    Learning Rule version of CGP.
    Contains a CGP graph calibrated to synaptic update
    """

    def __init__(self, parameters = None, *, 
                 n_rows: int = 1, n_cols: int = 1, 
                 function_list: List[Callable] = None,
                 arity: int = 2, prev_layer: int = None, seed: int = None,
                 learning_rate: float = 1.0, learning_rate_thr: float = 0.1, threshold_agg_func: Literal["max", "min", "mean", "sum"] = "mean",
                 delta_weight: bool = True, delta_threshold: bool = False,
                 use_trace_pre: bool = False, use_trace_post: bool = False, use_weights: bool = True, use_reward: bool = False, 
                 use_eligibility: bool = False, use_eligibility_pre: bool = False, use_eligibility_post: bool = False, use_eligibility_stdp: bool = False,
                 **kwargs):
        super().__init__(
            learning_rate=learning_rate, learning_rate_thr=learning_rate_thr,
            threshold_agg_func=threshold_agg_func, delta_weight=delta_weight, delta_threshold=delta_threshold,
            use_trace_pre=use_trace_pre, use_trace_post=use_trace_post,
            use_weights=use_weights, use_reward=use_reward,
            use_eligibility=use_eligibility, use_eligibility_pre=use_eligibility_pre, use_eligibility_post=use_eligibility_post,
            use_eligibility_stdp=use_eligibility_stdp
        )

        self.graph = CGP_Graph(
            n_inputs=self.input_size, n_outputs=self.output_size, n_rows=n_rows, n_cols=n_cols,
            genome=parameters,
            function_list=function_list, arity=arity, prev_layer=prev_layer, seed=seed
        )

    def forward(self, inp):
        return self.graph.forward(inp)

    @property
    def size(self):
        return self.graph.size
    
    @property
    def parameters(self):
        return self.graph.genome
    
    @parameters.setter
    def parameters(self, value):
        self.graph.genome = value