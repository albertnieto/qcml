import jax.numpy as jnp

def rbf_kernel(x1, x2, gamma=None):
    """
    Compute the RBF kernel using JAX.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Kernel value.
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[1]
    sq_dist = jnp.sum((x1[:, None] - x2[None, :]) ** 2, axis=-1)
    return jnp.exp(-gamma * sq_dist)
