from .jax.gaussian_kernel import gaussian_kernel as gaussian_kernel_jax
from .numpy.gaussian_kernel import gaussian_kernel as gaussian_kernel_numpy
from .torch.gaussian_kernel import gaussian_kernel as gaussian_kernel_torch

def gaussian_kernel(x1, x2, backend="numpy", gamma=1):
    """
    Compute the Gaussian kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.
        gamma (float): Scaling parameter.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return gaussian_kernel_jax(x1, x2, gamma)
    elif backend == "torch":
        return gaussian_kernel_torch(x1, x2, gamma)
    else:
        return gaussian_kernel_numpy(x1, x2, gamma)
