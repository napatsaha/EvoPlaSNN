import numpy as np


def target_func(x, y):
    return x**2 + y**2

def mse(a, b):
    soe = (a - b)**2
    return np.mean(soe)

def evaluate(solution, sample_size, num_inputs):
    inp = np.random.randn(num_inputs, sample_size)
    outp_bar = target_func(inp[0, :], inp[1,:])
    # outp_pred = solution.forward(inp)
    outp_pred = solution.forward(inp, squeeze=True)
    error = mse(outp_pred, outp_bar)
    return -error