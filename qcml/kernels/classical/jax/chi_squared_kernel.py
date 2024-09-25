import jax.numpy as jnp

def chi_squared_kernel(x1, x2):
    return jnp.sum((2 * x1 * x2) / (x1 + x2 + 1e-8))
