import numpy as np

def chi_squared_kernel(x1, x2):
    return np.sum((2 * x1 * x2) / (x1 + x2 + 1e-8))
