import jax.numpy as jnp

def histogram_intersection_kernel(x1, x2):
    return jnp.sum(jnp.minimum(x1, x2))
