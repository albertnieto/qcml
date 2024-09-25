from .jax.polynomial_kernel import polynomial_kernel as polynomial_kernel_jax
from .numpy.polynomial_kernel import polynomial_kernel as polynomial_kernel_numpy
from .torch.polynomial_kernel import polynomial_kernel as polynomial_kernel_torch

def polynomial_kernel(x1, x2, backend="numpy", degree=3, coef0=1):
    """
    Compute the polynomial kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.
        degree (int): Degree of the polynomial.
        coef0 (float): Independent term in polynomial kernel.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return polynomial_kernel_jax(x1, x2, degree, coef0)
    elif backend == "torch":
        return polynomial_kernel_torch(x1, x2, degree, coef0)
    else:
        return polynomial_kernel_numpy(x1, x2, degree, coef0)
