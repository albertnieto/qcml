import numpy as np

def laplacian_kernel(x1, x2, gamma=1):
    return np.exp(-gamma * np.sum(np.abs(x1 - x2)))
