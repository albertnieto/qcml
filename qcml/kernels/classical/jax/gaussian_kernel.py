import jax.numpy as jnp

def gaussian_kernel(x1, x2, gamma=1):
    return jnp.exp(-gamma * jnp.sum((x1 - x2) ** 2))
