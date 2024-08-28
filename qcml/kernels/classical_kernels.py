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

import numpy as np
import jax.numpy as jnp


def sigmoid_kernel(x, y, **params):
    """
    Compute the sigmoid kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.tanh(np.dot(x, y.T))


def laplacian_kernel(x, y, **params):
    """
    Compute the Laplacian kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.exp(-np.sum(np.abs(x[:, np.newaxis] - y), axis=-1))


def anova_kernel(x, y, D=2, **params):
    """
    Compute the ANOVA kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        D (int): Degree of the kernel. Default is 2.
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.prod(
        [np.sum(np.abs(x[:, np.newaxis] - y) ** d, axis=-1) for d in range(1, D + 1)],
        axis=0,
    )


def chi_squared_kernel(x, y, **params):
    """
    Compute the Chi-Squared kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    result = np.sum(
        (x[:, np.newaxis] - y) ** 2 / (x[:, np.newaxis] + y + 1e-9), axis=-1
    )
    result = np.clip(result, -700, 700)  # Clipping to avoid overflow in exp
    return np.exp(-result)


def histogram_intersection_kernel(x, y, **params):
    """
    Compute the Histogram Intersection kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.sum(np.minimum(x[:, np.newaxis], y), axis=-1)


def linear_kernel(x, y, **params):
    """
    Compute the Linear kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.dot(x, y.T)


def polynomial_kernel(x, y, degree=3, **params):
    """
    Compute the Polynomial kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        degree (int): Degree of the polynomial. Default is 3.
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return (np.dot(x, y.T) + 1) ** degree


def gaussian_kernel(x, y, **params):
    """
    Compute the Gaussian kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    return np.exp(-np.sum((x[:, np.newaxis] - y) ** 2, axis=-1))


def rbf_kernel(x, y, gamma=1.0, **params):
    """
    Compute the Radial Basis Function (RBF) kernel between two sets of vectors.

    Args:
        x (numpy.ndarray): First set of vectors with shape (n_samples_x, n_features).
        y (numpy.ndarray): Second set of vectors with shape (n_samples_y, n_features).
        gamma (float): Scaling parameter. Default is 1.0.
        **params: Additional kernel parameters (not used in this implementation).

    Returns:
        numpy.ndarray: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    result = -gamma * np.sum((x[:, np.newaxis] - y) ** 2, axis=-1)
    result = np.clip(result, -700, 700)  # Clipping to avoid overflow in exp
    return np.exp(result)
