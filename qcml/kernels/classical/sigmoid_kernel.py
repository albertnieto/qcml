from .jax.sigmoid_kernel import sigmoid_kernel as sigmoid_kernel_jax
from .numpy.sigmoid_kernel import sigmoid_kernel as sigmoid_kernel_numpy
from .torch.sigmoid_kernel import sigmoid_kernel as sigmoid_kernel_torch

def sigmoid_kernel(x1, x2, backend="numpy", alpha=1, c=0):
    """
    Compute the sigmoid kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.
        alpha (float): Scaling parameter.
        c (float): Bias term.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return sigmoid_kernel_jax(x1, x2, alpha, c)
    elif backend == "torch":
        return sigmoid_kernel_torch(x1, x2, alpha, c)
    else:
        return sigmoid_kernel_numpy(x1, x2, alpha, c)
