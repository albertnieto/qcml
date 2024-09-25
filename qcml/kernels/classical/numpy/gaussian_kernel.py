import numpy as np

def gaussian_kernel(x1, x2, gamma=1):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))
