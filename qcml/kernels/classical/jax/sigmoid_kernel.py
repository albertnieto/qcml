import jax.numpy as jnp

def sigmoid_kernel(x1, x2, alpha=1, c=0):
    return jnp.tanh(alpha * jnp.dot(x1, x2) + c)
