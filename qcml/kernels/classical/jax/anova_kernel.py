import jax.numpy as jnp

def anova_kernel(x1, x2, d=2, sigma=1):
    return jnp.sum(jnp.exp(-sigma * (x1 - x2) ** 2)) ** d
