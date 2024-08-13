import jax.numpy as jnp
from jax import vmap

def sigmoid_kernel(x1, x2, alpha=1, c=0):
    """
    Compute the sigmoid kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        alpha (float): Scaling parameter.
        c (float): Bias parameter.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.tanh(alpha * jnp.dot(x1, x2.T) + c)

def laplacian_kernel(x1, x2, gamma=1):
    """
    Compute the Laplacian kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.exp(-gamma * jnp.sum(jnp.abs(x1[:, None] - x2), axis=-1))

def anova_kernel(x1, x2, d=2, sigma=1):
    """
    Compute the ANOVA kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        d (int): Degree of the kernel.
        sigma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.sum(jnp.exp(-sigma * (x1[:, None] - x2)**2), axis=-1)**d

def chi_squared_kernel(x1, x2):
    """
    Compute the Chi-Squared kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.sum((2 * x1[:, None] * x2) / (x1[:, None] + x2 + 1e-8), axis=-1)

def histogram_intersection_kernel(x1, x2):
    """
    Compute the Histogram Intersection kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.sum(jnp.minimum(x1[:, None], x2), axis=-1)

def linear_kernel(x1, x2):
    """
    Compute the Linear kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.dot(x1, x2.T)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    """
    Compute the Polynomial kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        degree (int): Degree of the polynomial.
        coef0 (float): Independent term in polynomial kernel.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return (jnp.dot(x1, x2.T) + coef0)**degree

def gaussian_kernel(x1, x2, gamma=1):
    """
    Compute the Gaussian kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.exp(-gamma * jnp.sum((x1[:, None] - x2)**2, axis=-1))

def rbf_kernel(x1, x2, gamma=1):
    """
    Compute the Radial Basis Function (RBF) kernel between two sets of vectors.

    Args:
        x1 (jax.numpy.ndarray): First set of vectors with shape (n_samples_x1, n_features).
        x2 (jax.numpy.ndarray): Second set of vectors with shape (n_samples_x2, n_features).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Kernel matrix of shape (n_samples_x1, n_samples_x2).
    """
    return jnp.exp(-gamma * jnp.sum((x1[:, None] - x2)**2, axis=-1))
