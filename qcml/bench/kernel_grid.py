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

import itertools
from qcml.kernels.jax import (
    sigmoid_kernel,
    laplacian_kernel,
    anova_kernel,
    chi_squared_kernel,
    histogram_intersection_kernel,
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
    rbf_kernel,
)
from qcml.kernels import (
    separable_kernel,
    projected_quantum_kernel,
    iqp_embedding_kernel,
    angle_embedding_kernel,
    amplitude_embedding_kernel,
)

classical_kernel_param_map = {
    "sigmoid_kernel": {"alpha", "c"},
    "laplacian_kernel": {"gamma"},
    "anova_kernel": {"d", "sigma"},
    "chi_squared_kernel": set(),
    "histogram_intersection_kernel": set(),
    "linear_kernel": set(),
    "polynomial_kernel": {"degree", "coef0"},
    "gaussian_kernel": {"gamma"},
    "rbf_kernel": {"gamma"},
}


# Define the parameter grids for each kernel function
classical_kernel_grid = (
    [
        {"kernel_func": sigmoid_kernel, "kernel_params": {"alpha": alpha, "c": c}}
        for alpha, c in itertools.product([0.1, 0.5, 1.0], [0, 1, 5])
    ]
    + [
        {"kernel_func": laplacian_kernel, "kernel_params": {"gamma": gamma}}
        for gamma in [0.1, 1.0, 10.0]
    ]
    + [
        {"kernel_func": anova_kernel, "kernel_params": {"d": d, "sigma": sigma}}
        for d, sigma in itertools.product([2, 3], [0.1, 1.0])
    ]
    + [{"kernel_func": chi_squared_kernel, "kernel_params": {}}]
    + [{"kernel_func": histogram_intersection_kernel, "kernel_params": {}}]
    + [{"kernel_func": linear_kernel, "kernel_params": {}}]
    + [
        {
            "kernel_func": polynomial_kernel,
            "kernel_params": {"degree": degree, "coef0": coef0},
        }
        for degree, coef0 in itertools.product([2, 3, 4], [0, 1])
    ]
    + [
        {"kernel_func": gaussian_kernel, "kernel_params": {"gamma": gamma}}
        for gamma in [0.1, 1.0, 10.0]
    ]
    + [
        {"kernel_func": rbf_kernel, "kernel_params": {"gamma": gamma}}
        for gamma in [0.1, 1.0, 10.0]
    ]
)

reduced_classical_kernel_grid = (
    [
        {"kernel_func": sigmoid_kernel, "kernel_params": {"alpha": alpha, "c": c}}
        for alpha, c in itertools.product([0.1, 0.5, 1.0], [0, 1, 5])
    ]
    + [
        {
            "kernel_func": polynomial_kernel,
            "kernel_params": {"degree": degree, "coef0": coef0},
        }
        for degree, coef0 in itertools.product([2, 3, 4], [0, 1])
    ]
    + [
        {"kernel_func": rbf_kernel, "kernel_params": {"gamma": gamma}}
        for gamma in [0.1, 1.0, 10.0]
    ]
)

quantum_kernel_param_map = {
    "separable_kernel": {"encoding_layers"},
    "projected_quantum_kernel": {"trotter_steps", "t", "gamma_factor", "embedding"},
    "iqp_embedding_kernel": {"repeats"},
    "angle_embedding_kernel": {"rotation"},
    "amplitude_embedding_kernel": set(),
}


quantum_kernel_grid = (
    [
        {
            "kernel_func": projected_quantum_kernel,
            "kernel_params": {
                "trotter_steps": trotter_steps,
                "t": t,
                "gamma_factor": gamma_factor,
                "embedding": embedding,
            },
        }
        for trotter_steps, t, gamma_factor, embedding in itertools.product(
            [1, 3, 5], [0.01, 0.1, 1], [0.1, 1, 10], ["Hamiltonian", "IQP"]
        )
    ]
    + [
        {
            "kernel_func": separable_kernel,
            "kernel_params": {"encoding_layers": encoding_layers},
        }
        for encoding_layers in [1, 2, 3]
    ]
    + [
        {"kernel_func": iqp_embedding_kernel, "kernel_params": {"repeats": repeats}}
        for repeats in [1, 2, 3]
    ]
    + [
        {"kernel_func": angle_embedding_kernel, "kernel_params": {"rotation": rotation}}
        for rotation in ["X", "Y", "Z"]
    ]
    + [{"kernel_func": amplitude_embedding_kernel, "kernel_params": {}}]
)

kernel_grid = classical_kernel_grid + quantum_kernel_grid
kernel_param_map = {**classical_kernel_param_map, **quantum_kernel_param_map}

reduced_quantum_kernel_grid = (
    [
        {
            "kernel_func": projected_quantum_kernel,
            "kernel_params": {
                "trotter_steps": trotter_steps,
                "t": t,
                "gamma_factor": gamma_factor,
            },
        }
        for trotter_steps, t, gamma_factor in itertools.product(
            [1, 3, 5], [0.01, 0.1, 1], [0.1, 1, 10]
        )
    ]
    + [
        {"kernel_func": iqp_embedding_kernel, "kernel_params": {"repeats": repeats}}
        for repeats in [1, 2, 3]
    ]
)
reduced_kernel_grid = (
    [
        {
            "kernel_func": projected_quantum_kernel,
            "kernel_params": {
                "trotter_steps": trotter_steps,
                "t": t,
                "gamma_factor": gamma_factor,
            },
        }
        for trotter_steps, t, gamma_factor in itertools.product(
            [1, 3, 5], [0.01, 0.1, 1], [0.1, 1, 10]
        )
    ]
    + [
        {
            "kernel_func": separable_kernel,
            "kernel_params": {"encoding_layers": encoding_layers},
        }
        for encoding_layers in [1, 2, 3]
    ] + [
        {"kernel_func": rbf_kernel, "kernel_params": {"gamma": gamma}}
        for gamma in [0.1, 1.0, 10.0]
    ]
)
more_reduced_kernel_grid = reduced_classical_kernel_grid + reduced_quantum_kernel_grid

def select_kernels(selected_kernels, kernel_grid=kernel_grid):
    """
    Filter the kernel grid to include only selected kernels.

    Parameters:
    - kernel_grid (list): The complete list of kernel configurations.
    - selected_kernels (list): The list of kernel function names to include.

    Returns:
    - filtered_grid (list): The filtered list of kernel configurations.
    """
    filtered_grid = [
        kernel
        for kernel in kernel_grid
        if kernel["kernel_func"].__name__ in selected_kernels
    ]
    return filtered_grid
