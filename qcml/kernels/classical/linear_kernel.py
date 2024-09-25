from .jax.linear_kernel import linear_kernel as linear_kernel_jax
from .numpy.linear_kernel import linear_kernel as linear_kernel_numpy
from .torch.linear_kernel import linear_kernel as linear_kernel_torch

def linear_kernel(x1, x2, backend="numpy"):
    """
    Compute the linear kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return linear_kernel_jax(x1, x2)
    elif backend == "torch":
        return linear_kernel_torch(x1, x2)
    else:
        return linear_kernel_numpy(x1, x2)
