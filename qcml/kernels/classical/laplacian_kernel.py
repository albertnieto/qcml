from .jax.laplacian_kernel import laplacian_kernel as laplacian_kernel_jax
from .numpy.laplacian_kernel import laplacian_kernel as laplacian_kernel_numpy
from .torch.laplacian_kernel import laplacian_kernel as laplacian_kernel_torch

def laplacian_kernel(x1, x2, backend="numpy", gamma=1):
    """
    Compute the Laplacian kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.
        gamma (float): Scaling parameter.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return laplacian_kernel_jax(x1, x2, gamma)
    elif backend == "torch":
        return laplacian_kernel_torch(x1, x2, gamma)
    else:
        return laplacian_kernel_numpy(x1, x2, gamma)
