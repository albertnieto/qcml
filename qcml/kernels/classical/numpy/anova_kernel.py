import numpy as np

def anova_kernel(x1, x2, d=2, sigma=1):
    return np.sum(np.exp(-sigma * (x1 - x2) ** 2)) ** d
