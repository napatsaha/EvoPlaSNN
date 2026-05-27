import numpy as np
from common.base import Evaluator


def rastrigin_nD(*x, A=10, offsets=None):
    """Rastrigin function."""
    n = len(x)
    if offsets is None:
        offsets = [0] * n
    assert len(offsets) == n, "Offsets must match the number of dimensions."
    return A*n + sum([(xi - oi)**2 - A * np.cos(2 * np.pi * (xi - oi)) for xi, oi in zip(x, offsets)])


class SimpleRastriginFunc(Evaluator):
    def __init__(self, ndim=2, A=10, offsets=None):
        super().__init__()
        self.ndim = ndim
        self.A = A
        self.offsets = offsets if offsets is not None else [0] * ndim

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rastrigin function at point x.
        
        :param x: A numpy array of shape (ndim,) representing the input point.
        :return: The value of the Rastrigin function at point x.
        """
        assert len(x) == self.ndim, "Input dimension must match the evaluator's ndim."
        return rastrigin_nD(*x, A=self.A, offsets=self.offsets)