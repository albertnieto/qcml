# Copyright 2024 Albert Nieto

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp


def sigmoid_kernel(x1, x2, alpha=1, c=0):
    """
    Compute the sigmoid kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        alpha (float): Scaling parameter.
        c (float): Bias parameter.

    Returns:
        jax.numpy.ndarray: Scalar value representing the sigmoid kernel.
    """
    return jnp.tanh(alpha * jnp.dot(x1, x2) + c)


def laplacian_kernel(x1, x2, gamma=1):
    """
    Compute the Laplacian kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Scalar value representing the Laplacian kernel.
    """
    return jnp.exp(-gamma * jnp.sum(jnp.abs(x1 - x2)))


def anova_kernel(x1, x2, d=2, sigma=1):
    """
    Compute the ANOVA kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        d (int): Degree of the kernel.
        sigma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Scalar value representing the ANOVA kernel.
    """
    return jnp.sum(jnp.exp(-sigma * (x1 - x2) ** 2)) ** d


def chi_squared_kernel(x1, x2):
    """
    Compute the Chi-Squared kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).

    Returns:
        jax.numpy.ndarray: Scalar value representing the Chi-Squared kernel.
    """
    return jnp.sum((2 * x1 * x2) / (x1 + x2 + 1e-8))


def histogram_intersection_kernel(x1, x2):
    """
    Compute the Histogram Intersection kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).

    Returns:
        jax.numpy.ndarray: Scalar value representing the Histogram Intersection kernel.
    """
    return jnp.sum(jnp.minimum(x1, x2))


def linear_kernel(x1, x2):
    """
    Compute the Linear kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).

    Returns:
        jax.numpy.ndarray: Scalar value representing the Linear kernel.
    """
    return jnp.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=3, coef0=1):
    """
    Compute the Polynomial kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        degree (int): Degree of the polynomial.
        coef0 (float): Independent term in polynomial kernel.

    Returns:
        jax.numpy.ndarray: Scalar value representing the Polynomial kernel.
    """
    return (jnp.dot(x1, x2) + coef0) ** degree


def gaussian_kernel(x1, x2, gamma=1):
    """
    Compute the Gaussian kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Scalar value representing the Gaussian kernel.
    """
    return jnp.exp(-gamma * jnp.sum((x1 - x2) ** 2))


def rbf_kernel(x1, x2, gamma=1):
    """
    Compute the Radial Basis Function (RBF) kernel between two vectors.

    Args:
        x1 (jax.numpy.ndarray): First vector with shape (n_features,).
        x2 (jax.numpy.ndarray): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        jax.numpy.ndarray: Scalar value representing the RBF kernel.
    """
    return jnp.exp(-gamma * jnp.sum((x1 - x2) ** 2))
