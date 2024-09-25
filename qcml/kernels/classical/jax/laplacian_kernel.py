import jax.numpy as jnp

def laplacian_kernel(x1, x2, gamma=1):
    return jnp.exp(-gamma * jnp.sum(jnp.abs(x1 - x2)))
