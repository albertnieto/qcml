import numpy as np

def rbf_kernel(x1, x2, gamma=None):
    """
    Compute the RBF kernel using NumPy.

    Args:
        x1 (numpy.ndarray): First vector with shape (n_features,).
        x2 (numpy.ndarray): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        numpy.ndarray: Kernel value.
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[1]
    sq_dist = np.sum((x1[:, None] - x2[None, :]) ** 2, axis=-1)
    return np.exp(-gamma * sq_dist)
