import jax.numpy as jnp

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (jnp.dot(x1, x2) + coef0) ** degree
