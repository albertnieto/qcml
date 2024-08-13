import jax
import jax.numpy as jnp
import numpy as np
import logging
from qic.utils.model import chunk_vmapped_fn

logger = logging.getLogger(__name__)

def compute_gram_matrix(self, X1, X2, kernel_func, max_vmap=250, **kernel_params):
    """
    Compute the kernel matrix relative to data sets X1 and X2 using a given kernel function.

    Args:
        X1 (np.array): First dataset of input vectors.
        X2 (np.array): Second dataset of input vectors.
        kernel_func (callable): Kernel function to be applied.
        kernel_params (dict): Additional parameters for the kernel function.
        max_vmap (int or None): The maximum size of a chunk to vectorize over. Lower values use less memory.

    Returns:
        kernel_matrix (np.array): Matrix of size (len(X1), len(X2)) with elements K(x_1, x_2).
    """
    dim1 = len(X1)
    dim2 = len(X2)

    # Concatenate all pairs of vectors
    Z = jnp.array(
        [np.concatenate((X1[i], X2[j])) for i in range(dim1) for j in range(dim2)]
    )

    # Use kernel function instead of circuit
    if kernel_params is None:
        kernel_params = {}

    logger.info(f'Gram matrix of {kernel_func.__name} with parameters {kernel_params}')
    
    # Vectorize the kernel function over the concatenated pairs
    batched_kernel_func = chunk_vmapped_fn(
        jax.vmap(lambda z: kernel_func(z[:len(X1[0])], z[len(X1[0]):], **kernel_params), 0),
        start=0, max_vmap=max_vmap
    )
    
    kernel_values = batched_kernel_func(Z)

    # Reshape the values into the kernel matrix
    kernel_matrix = np.reshape(kernel_values, (dim1, dim2))

    return kernel_matrix
