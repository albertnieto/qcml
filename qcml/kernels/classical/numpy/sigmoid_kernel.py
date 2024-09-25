import numpy as np

def sigmoid_kernel(x1, x2, alpha=1, c=0):
    return np.tanh(alpha * np.dot(x1, x2) + c)
