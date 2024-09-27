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

import pytest
import jax.numpy as jnp
import numpy as np
import torch
from qcml.kernels.classical import (
    rbf_kernel,
    sigmoid_kernel,
    laplacian_kernel,
    anova_kernel,
    chi_squared_kernel,
    histogram_intersection_kernel,
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
)

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

# Convert data to different backend formats
x1_jax = jnp.array(x1)
x2_jax = jnp.array(x2)
x1_torch = torch.tensor(x1, dtype=torch.float32)
x2_torch = torch.tensor(x2, dtype=torch.float32)


# Check if result is scalar or array depending on the kernel type
def is_scalar_or_array(result, backend):
    if backend == "numpy":
        return isinstance(result, (np.ndarray, np.generic))
    elif backend == "jax":
        return isinstance(result, (jnp.ndarray, float))
    elif backend == "torch":
        return isinstance(result, (torch.Tensor, float))


# RBF Kernel
def test_rbf_kernel_numpy():
    result = rbf_kernel(x1, x2, backend="numpy")
    assert isinstance(result, np.ndarray)


def test_rbf_kernel_jax():
    result = rbf_kernel(x1_jax, x2_jax, backend="jax")
    assert isinstance(result, jnp.ndarray)


def test_rbf_kernel_torch():
    result = rbf_kernel(x1_torch, x2_torch, backend="torch")
    assert isinstance(result, torch.Tensor)


# Sigmoid Kernel
def test_sigmoid_kernel_numpy():
    result = sigmoid_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_sigmoid_kernel_jax():
    result = sigmoid_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_sigmoid_kernel_torch():
    result = sigmoid_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Laplacian Kernel
def test_laplacian_kernel_numpy():
    result = laplacian_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_laplacian_kernel_jax():
    result = laplacian_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_laplacian_kernel_torch():
    result = laplacian_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Anova Kernel
def test_anova_kernel_numpy():
    result = anova_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_anova_kernel_jax():
    result = anova_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_anova_kernel_torch():
    result = anova_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Chi-Squared Kernel
def test_chi_squared_kernel_numpy():
    result = chi_squared_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_chi_squared_kernel_jax():
    result = chi_squared_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_chi_squared_kernel_torch():
    result = chi_squared_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Histogram Intersection Kernel
def test_histogram_intersection_kernel_numpy():
    result = histogram_intersection_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_histogram_intersection_kernel_jax():
    result = histogram_intersection_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_histogram_intersection_kernel_torch():
    result = histogram_intersection_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Linear Kernel
def test_linear_kernel_numpy():
    result = linear_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_linear_kernel_jax():
    result = linear_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_linear_kernel_torch():
    result = linear_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Polynomial Kernel
def test_polynomial_kernel_numpy():
    result = polynomial_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_polynomial_kernel_jax():
    result = polynomial_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_polynomial_kernel_torch():
    result = polynomial_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")


# Gaussian Kernel
def test_gaussian_kernel_numpy():
    result = gaussian_kernel(x1, x2, backend="numpy")
    assert is_scalar_or_array(result, "numpy")


def test_gaussian_kernel_jax():
    result = gaussian_kernel(x1_jax, x2_jax, backend="jax")
    assert is_scalar_or_array(result, "jax")


def test_gaussian_kernel_torch():
    result = gaussian_kernel(x1_torch, x2_torch, backend="torch")
    assert is_scalar_or_array(result, "torch")
