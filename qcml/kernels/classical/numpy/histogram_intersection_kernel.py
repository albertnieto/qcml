import numpy as np

def histogram_intersection_kernel(x1, x2):
    return np.sum(np.minimum(x1, x2))
