import pytest
import jax.numpy as jnp
import numpy as np
import torch
from classical import (
    rbf_kernel,
    sigmoid_kernel,
    laplacian_kernel,
    anova_kernel,
    chi_squared_kernel,
    histogram_intersection_kernel,
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel
)

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

# Convert data to different backend formats
x1_jax = jnp.array(x1)
x2_jax = jnp.array(x2)
x1_torch = torch.tensor(x1)
x2_torch = torch.tensor(x2)

def test_rbf_kernel_numpy():
    result = rbf_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_rbf_kernel_jax():
    result = rbf_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_rbf_kernel_torch():
    result = rbf_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_sigmoid_kernel_numpy():
    result = sigmoid_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_sigmoid_kernel_jax():
    result = sigmoid_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_sigmoid_kernel_torch():
    result = sigmoid_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_laplacian_kernel_numpy():
    result = laplacian_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_laplacian_kernel_jax():
    result = laplacian_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_laplacian_kernel_torch():
    result = laplacian_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_anova_kernel_numpy():
    result = anova_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_anova_kernel_jax():
    result = anova_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_anova_kernel_torch():
    result = anova_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_chi_squared_kernel_numpy():
    result = chi_squared_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_chi_squared_kernel_jax():
    result = chi_squared_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_chi_squared_kernel_torch():
    result = chi_squared_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_histogram_intersection_kernel_numpy():
    result = histogram_intersection_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_histogram_intersection_kernel_jax():
    result = histogram_intersection_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_histogram_intersection_kernel_torch():
    result = histogram_intersection_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_linear_kernel_numpy():
    result = linear_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_linear_kernel_jax():
    result = linear_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_linear_kernel_torch():
    result = linear_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_polynomial_kernel_numpy():
    result = polynomial_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_polynomial_kernel_jax():
    result = polynomial_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_polynomial_kernel_torch():
    result = polynomial_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)

def test_gaussian_kernel_numpy():
    result = gaussian_kernel(x1, x2, backend='numpy')
    assert isinstance(result, np.ndarray)

def test_gaussian_kernel_jax():
    result = gaussian_kernel(x1_jax, x2_jax, backend='jax')
    assert isinstance(result, jnp.ndarray)

def test_gaussian_kernel_torch():
    result = gaussian_kernel(x1_torch, x2_torch, backend='torch')
    assert isinstance(result, torch.Tensor)
