import numpy as np


def softmax(x, axis=0):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def cross_entropy_loss(p, q, axis=0, epsilon=1e-12):
    return -(p * np.log(q+epsilon)).sum(axis=axis)

def mean_square_error(p, q, axis=0):
    se = np.square(p - q)
    return np.mean(se, axis=axis, where=~np.isnan(se))

def safe_divide(x, y, default_value=np.inf):
    """Safe division that avoids division by zero."""
    return np.divide(x, y, where=y != 0, out=np.full(np.broadcast(x, y).shape, default_value, dtype=float))