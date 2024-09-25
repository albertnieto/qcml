import jax.numpy as jnp

def linear_kernel(x1, x2):
    return jnp.dot(x1, x2)
