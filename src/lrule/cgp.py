"""
Cartesian Genetic Programming: graph program, and learning rule
"""

from lrule.utils import tile_array
import numpy as np
from numpy.typing import ArrayLike
from common.base import LearningRule, SynapseLayerProtocol
from lrule.base import BaseLearningRule
from evo.base import Genome
from typing import List, Tuple, Literal, Callable
import copy

from enum import IntEnum

FUNCTION_LIST = [
    np.add,
    np.subtract,
    np.multiply,
    np.divide
]


class CGP_Graph:
    GENE_TYPE = IntEnum('GeneType', ["Connection", "Function", "Output"])

    def __init__(self, 
                 n_inputs: int, n_outputs: int, n_rows: int, n_cols: int, *,
                 genome: list = None, function_list: List[Callable] = None,
                 arity: int = 2, levels_back: int = None, allow_input_anywhere: bool = False,
                 mutation_rate: int | float = 0.1, mu_c: float = None, mu_f: float = None, mu_o: float = None,
                 mutation_method: Literal["point", "prob"] = "point",
                 seed: int = None):
        self._ni = n_inputs
        self._no = n_outputs
        self._nr = n_rows
        self._nc = n_cols
        self._nn = self._nr * self._nc
        self._a = arity
        self._l = levels_back if levels_back is not None else self._nc
        self._allow_inputs = allow_input_anywhere

        # Allele length
        self._al = self._a + 1
        # Genome length
        self._lg = self._nn * self._al + self._no
        # Maximum nodes (inputs + internal)
        self._m = self._ni + self._nn
        # Maxmimum nodes (inputs + internal + outputs)
        self._mo = self._ni + self._nn + self._no

        # functions
        self.function_list = []
        if function_list is None:
            self.function_list = FUNCTION_LIST
        else:
            # Check if passed in function is a callable
            for f in function_list:
                if isinstance(f, Callable):
                    self.function_list.append(f)
                elif isinstance(f, str):
                    # Try getting function from numpy
                    try:
                        f = getattr(np, f)
                    except:
                        raise Exception(f"Could not find function \'{f}\' from numpy")
                    self.function_list.append(f)

        # self.function_list = FUNCTION_LIST if function_list is None else function_list
        self._nf = len(self.function_list)

        # Mutation
        self.mutation_method = mutation_method
        if mutation_method == "point":
            if isinstance(mutation_rate, float) and (0 <= mutation_rate < 1.0):
                self._mu_min = int(mutation_rate * self._lg)
                self._mu_rate = mutation_rate
            elif isinstance(mutation_rate, int) and (1 <= mutation_rate <= self._lg):
                self._mu_min = int(mutation_rate)
                self._mu_rate = mutation_rate / self._lg
            else:
                raise ValueError(f"Invalid mutation rate. Must be either fraction [0, 1) or an integer between [0, {self._lg}]")
        # TODO: Add probabilistic mutation
        elif mutation_method == "prob":
            raise NotImplementedError("Probabilistic mutation not yet implemented")
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
        self._node_outputs = np.zeros((self._mo, 1), dtype=np.float64)

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

    def _mutate(self) -> np.ndarray[int]:
        # TODO: Add probabilistic mutation
        # For now, assume point mutation
        
        # Find gene id to mutate based on _mu_min
        gene_to_mutate = np.random.randint(self._lg, size=(self._mu_min, ))

        for gene in gene_to_mutate:
            gene_type = self._determine_gene_type(gene)
            # Output genes
            if gene_type == self.GENE_TYPE.Output:
                idx = gene - (self._nn * self._al)
                self._genome_output[idx] = np.random.randint(self._no)
            # Function genes
            elif gene_type == self.GENE_TYPE.Function:
                idx = self.flat_to_coord_3d(gene)
                self._genome_internal[idx] = np.random.randint(self._nf)
            # Connection gene
            elif gene_type == self.GENE_TYPE.Connection:
                i, j, k = self.flat_to_coord_3d(gene)
                valid_genes = self._find_permissable_connection_genes(j)
                self._genome_internal[i, j, k] = np.random.choice(valid_genes)
            else:
                raise ValueError(f"Gene ID {gene} with Type: {gene_type} not valid.")
            
        return gene_to_mutate

    def mutate(self) -> "CGP_Graph":
        """
        Mutate current graph and produce a new copy of an offspring.
        """
        offspring = copy.deepcopy(self)
        offspring._mutate()
        offspring._find_active_nodes()
        return offspring

    def _create_random_genome(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        # Randomise connection genes
        for j in range(self._nc):
            req_shape = self._genome_internal[:, j, :self._a].shape
            if self._allow_inputs:
                permissables = self._find_permissable_connection_genes(j)
                self._genome_internal[:, j, :self._a] = np.random.choice(permissables, size=req_shape)
            else:
                upper_bound, lower_bound = self._get_column_bounds(j)
                self._genome_internal[:, j, :self._a] = np.random.randint(lower_bound, upper_bound, size=req_shape)
        # Randomise function genes
        req_shape = self._genome_internal[:, :, self._a].shape
        self._genome_internal[:, :, self._a] = np.random.randint(0, self._nf, size=req_shape)
        # Randomise output genes
        self._genome_output[:] = np.random.randint(0, self._m, size=self._genome_output.shape)
        

    def _get_column_bounds(self, j) -> tuple:
        upper_bound = self._ni + j*self._nr
        lower_bound = 0 if j < self._l else self._ni + (j - self._l)*self._nr
        return upper_bound, lower_bound
    
    def _find_permissable_connection_genes(self, j) -> np.ndarray:
        """
        Returns permissable input nodes into connection genes given a column id, *j*.  
        If `allow_inputs_anywhere = True`, also includes input node id's.
        """
        upper_bound, lower_bound = self._get_column_bounds(j)
        permissables = np.arange(lower_bound, upper_bound)
        if self._allow_inputs and j >= self._l:
            # Appends input nodes to permissable genes
            permissables = np.concatenate([np.arange(self._ni), permissables])
        return permissables
        
    def _determine_gene_type(self, gene: int) -> int:
        """
        Determine whether gene is a connection (0), function (1) or output (2) gene.
        """
        assert 0 <= gene <= self._lg, f"Gene id: {gene} is outside of valid range [0, {self._lg}]"
        # Output gene
        if gene >= (self._nn * self._al):
            return self.GENE_TYPE.Output
        elif gene % self._al == self._a:
            return self.GENE_TYPE.Function
        else:
            return self.GENE_TYPE.Connection

    def _fill_genome(self, genome: list):
        """
        When a genome is given, fill it into `_genome_internal` and `_genome_output` properly
        """
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
                upper_bound, lower_bound = self._get_column_bounds(j)
                # upper_bound = self._ni + j*self._nr
                # if j >= self._l:
                #     lower_bound = self._ni + (j - self._l)*self._nr
                #     # assert np.all((self._ni + (j - self._l)*self._nr) <= c[:, j]) & np.all(c[:, j] < (self._ni + j*self._nr)), f"Connection gene invalid at column {j}"
                # else:
                #     lower_bound = 0
                #     # assert np.all(0 <= c[:, j]) & np.all(c[:, j] < (self._ni + j*self._nr)), f"Connection gene invalid at column {j}"
                assert np.all(lower_bound <= c[:, j]) & np.all(c[:, j] < upper_bound), f"Connection gene invalid at column {j}"

    def _find_active_nodes(self):
        self._active_nodes = []
        nodes_to_test = [] # Create a separate array to test to avoid including input nodes in final array
        nodes_to_test.extend(self._genome_output)
        for oi in nodes_to_test:
            if (oi - self._ni) < 0:
                # Ignore if input node
                continue
            else:
                if oi not in self._active_nodes: # Avoid double counting
                    self._active_nodes.append(oi)
                i, j = self.flat_to_coord(oi - self._ni)
                gene = self._genome_internal[i, j, :]
                # Check each connection genes in this node
                for conn_gene in gene[:-1]:
                    if (conn_gene - self._ni) < 0:
                        # Ignore if input node
                        continue
                    if conn_gene not in nodes_to_test:
                        self._active_nodes.append(conn_gene)
                        nodes_to_test.append(conn_gene)
        self._active_nodes = np.sort(np.array(self._active_nodes))
        
    def flat_to_coord(self, idx: int) -> tuple:
        if idx < 0:
            raise Warning(f"Flat index {idx} should be non-negative integer")
        i = idx % self._nr
        j = idx // self._nr
        return i, j
    
    def flat_to_coord_3d(self, idx: int) -> tuple:
        """
        Convert single genome index to 3-dim indices
        """
        if idx < 0:
            raise Warning(f"Flat index {idx} should be non-negative integer")
        k = idx % self._al
        id2d = idx // self._al
        i = id2d % self._nr
        j = id2d // self._nr
        return i, j, k

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

    def _tuplify_genome(self) -> str:
        internals = self._genome_internal.swapaxes(0, 1).reshape(-1, self._al).tolist()
        outputs = self._genome_output.tolist()
        return str(internals + outputs)

    def __getitem__(self, index):
        if not index >= self._ni:
            raise IndexError(f"Index values must be >= number of inputs {self._ni}. Got {index}")
        if not index < self._mo:
            raise IndexError(f"Index value must be < total number of nodes {self._mo}. Got {index}")
        if index < self._m:
            i, j = self.flat_to_coord(index - self._ni)
            return self._genome_internal[i, j, :]
        elif index >= self._m:
            return self._genome_output[index - self._m]
        else:
            raise IndexError(f"Invalid index {index}")

    def __str__(self):
        s = "CGP Graph (active nodes only)" + "\n"
        s += "Input Nodes: " + str([*range(self._ni)]) + "\n"
        for node in self.active_nodes:
            s += f"Node {node}: " + str(self[node]) + "\n"
        for oi, o in enumerate(self._genome_output):
            s+= f"Output {oi}: Node " + str(o) + "\n"
        s = s.rstrip("\n")
        return s


    def __repr__(self):
        return f"CGP_Graph({self._tuplify_genome()}, n_inputs={self.n_inputs}, n_rows={self.n_rows}, n_cols={self.n_cols}, n_outputs={self.n_outputs})"


class CGP_Rule(BaseLearningRule, Genome):
    """
    Learning Rule version of CGP.
    Contains a CGP graph calibrated to synaptic update
    """
    graph: CGP_Graph

    def __init__(self, parameters = None, *, 
                 n_rows: int = 1, n_cols: int = 1, 
                 function_list: List[Callable] = None,
                 arity: int = 2, levels_back: int = None, allow_input_anywhere: bool = False,
                 mutation_rate: int | float = 0.1, mu_c: float = None, mu_f: float = None, mu_o: float = None,
                 mutation_method: Literal["point", "prob"] = "point",
                 seed: int = None,
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
            function_list=function_list, arity=arity, 
            levels_back=levels_back, allow_input_anywhere=allow_input_anywhere,
            mutation_method=mutation_method, mutation_rate=mutation_rate, mu_c=mu_c, mu_f=mu_f, mu_o=mu_o,
            seed=seed,
            **kwargs
        )

    def forward(self, inp):
        return self.graph.forward(inp)

    def mutate(self) -> 'CGP_Rule':
        new_graph = self.graph.mutate()
        new_rule = copy.copy(self)
        new_rule.graph = new_graph
        return new_rule

    @property
    def size(self):
        return self.graph.size
    
    @property
    def parameters(self):
        return self.graph.genome
    
    @parameters.setter
    def parameters(self, value):
        self.graph.genome = value

    # Alias
    genome = parameters