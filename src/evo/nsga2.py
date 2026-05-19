"""
2026-05-19
NSGA-II
Assuming 2 objectives for now
"""

from common.base import Solver
import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def dominates(sol1, sol2, obj1, obj2):
    """
    Check whether one solution dominates another, based on 2 objectives

    Args:
        sol1 (_type_): Index of Solution 1
        sol2 (_type_): Index of SOlution 2
        obj1 (_type_): List of objective 1 values
        obj2 (_type_): List of objective 2 values

    Returns:
        bool: Whether sol1 dominates sol2 in at least one objective, and is not worse than sol2 in both objectives
    """
    not_worse_than = (obj1[sol1] <= obj1[sol2]) and (obj2[sol1] <= obj2[sol2])
    at_least_one_better = (obj1[sol1] < obj1[sol2]) or (obj2[sol1] < obj2[sol2])
    return not_worse_than and at_least_one_better
# def f1(x):
#     return np.sum(x, axis=1)

# def f2(x):
#     return np.sum(x**2, axis=1)

def fast_nondominated_sort(sols: np.ndarray, obj1, obj2):
    N = len(obj1)
    S_dom_by = [[] for _ in range(N)]
    n_dom_over = np.zeros(shape=N)
    ranks = np.empty(shape=N)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            # if p == q:
            #     continue
            if dominates(p, q, obj1, obj2):
                S_dom_by[p].append(q)
            elif dominates(q, p, obj1, obj2):
                n_dom_over[p] += 1
        if n_dom_over[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S_dom_by[p]:
                n_dom_over[q] -= 1
                if n_dom_over[q] == 0:
                    ranks[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts, ranks


def plot_fronts(fronts, obj1, obj2, cmap="tab10"):

    cmap = plt.colormaps.get_cmap(cmap)

    for i, F in enumerate(fronts):
        if len(F) == 0:
            continue
        for sol in F:
            plt.plot(obj1[sol], obj2[sol], '.', label=i, color=cmap(i))
            plt.text(obj1[sol], obj2[sol], sol, color=cmap(i), size=12, va="top", ha="center")
    # plt.legend()
    plt.show()


def crowding_distance(fronts, N, obj1, obj2):
    # N = len(obj1)
    dist = np.zeros(shape=N)

    # I = fronts[0]
    for I in fronts:
        l = len(I)
        if l == 0: continue
        for obj in (obj1, obj2):
            obj_f = np.take(obj, I)
            I_sorted = np.take(I, np.argsort(obj_f))
            fm_min = min(obj_f)
            fm_max = max(obj_f)
            dist[I_sorted[0]] = np.inf
            dist[I_sorted[-1]] = np.inf
            for i in range(1, l-1):
                idx = I_sorted[i]
                dist[idx] += (obj[I_sorted[i+1]] - obj[I_sorted[i-1]]) / (fm_max - fm_min)

    return dist


def crowding_distance_single_front(front, N, obj1, obj2):
    """
    For only one front
    """
    dist = np.zeros(shape=N)

    I = front
    l = len(I)
    for obj in (obj1, obj2):
        obj_f = np.take(obj, I)
        I_sorted = np.take(I, np.argsort(obj_f))
        fm_min = min(obj_f)
        fm_max = max(obj_f)
        dist[I_sorted[0]] = np.inf
        dist[I_sorted[-1]] = np.inf
        for i in range(1, l-1):
            idx = I_sorted[i]
            dist[idx] += (obj[I_sorted[i+1]] - obj[I_sorted[i-1]]) / (fm_max - fm_min)

    return dist


def crowded_compare(sol1, sol2, ranks, dist):
    """
    Comparison between two solutions based on Pareto front rank (ascending) and crowding distance (descending)
    """
    if ranks[sol1] < ranks[sol2]:
        return True
    elif ranks[sol1] == ranks[sol2]:
        return dist[sol1] > dist[sol2]
    else:
        return False


def select_parents(pop, N, ranks, dist):
    # N = pop.shape[0]
    parents = []
    for i in range(2):
        ix, iy = np.random.choice(N, size=2, replace=False)
        if crowded_compare(ix, iy, ranks, dist):
            parents.append(pop[ix])
        else:
            parents.append(pop[iy])
    return parents


def crossover(x, y, p):
    assert (len(x) == len(y))
    N = len(x)
    p = np.clip(p, 0, 1)
    r = np.random.binomial(1, p, N)
    offspring = np.where(r, x, y)
    return offspring


def mutate(x, p):
    N = len(x)
    r = np.random.binomial(1, p, N)
    delta = np.random.normal(0, 0.1, N)
    return np.where(r, x + delta, x)



class NSGA2:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.P = None
        self.P_o1 = None
        self.P_o2 = None
        self.Q = None
        self.Q_o1 = None
        self.Q_o2 = None

    def ask(self, g):
        if g == 0:
            sols = np.random.randn(self.N, self.D)
            self.P = sols
        else:             
            sols = self.Q
        return sols
    
    def tell(self, o1, o2, g):
        if g == 0:
            self.P_o1 = o1
            self.P_o2 = o2

        else:
            self.Q_o1 = o1
            self.Q_o2 = o2
            # Next generation
            R = np.concatenate([self.P, self.Q], axis=0)
            o1_R = np.concatenate([self.P_o1, self.Q_o1])
            o2_R = np.concatenate([self.P_o2, self.Q_o2])

            fronts_R, ranks_R = fast_nondominated_sort(R, o1_R, o2_R)

            self.P, self.P_o1, self.P_o2 = self.select_new_parents(R, fronts_R, ranks_R, o1_R, o2_R)


        fronts_P, ranks_P = fast_nondominated_sort(self.P, self.P_o1, self.P_o2)
        dist_P = crowding_distance(fronts_P, self.N, self.P_o1, self.P_o2)

        self.Q = self.create_offspring(self.P, ranks_P, dist_P)

    def select_new_parents(self, R, fronts_R, ranks_R, o1_R, o2_R):
        idx_parents = []

        n = 0
        for f in fronts_R:
            n_front = len(f)
            # n += n_front
            if n + n_front <= self.N:
                idx_parents.extend(f)
                n += n_front
            else:
                dist = crowding_distance_single_front(f, R.shape[0], o1_R, o2_R)
                dist_f = np.take(dist, f)
                sorted_f = np.take(f, np.argsort(-dist_f))
                idx_parents.extend(sorted_f[:(self.N - n)])
                break
    
        P = np.take(R, idx_parents, axis=0)
        o1 = np.take(o1_R, idx_parents)
        o2 = np.take(o2_R, idx_parents)

        return P, o1, o2

    def create_offspring(self, P, ranks_P, dist_P):
        # Creating offspring
        Q = []
        for i in range(self.N):
            px, py = select_parents(P, self.N, ranks_P, dist_P)
            off = crossover(px, py, p=0.5)
            off = mutate(off, p=0.5)
            Q.append(off)
        Q = np.stack(Q, axis=0)
        return Q