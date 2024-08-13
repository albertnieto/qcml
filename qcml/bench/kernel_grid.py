import itertools
from qic.kernels.jax import (
    sigmoid_kernel,
    laplacian_kernel,
    anova_kernel,
    chi_squared_kernel,
    histogram_intersection_kernel,
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
    rbf_kernel
)
from qic.kernels import (
    iqp_kernel,
    angle_embedding_kernel,
    amplitude_embedding_kernel,
)

classical_kernel_param_map = {
    'sigmoid_kernel': {'alpha', 'c'},
    'laplacian_kernel': {'gamma'},
    'anova_kernel': {'d', 'sigma'},
    'chi_squared_kernel': set(),
    'histogram_intersection_kernel': set(),
    'linear_kernel': set(),
    'polynomial_kernel': {'degree', 'coef0'},
    'gaussian_kernel': {'gamma'},
    'rbf_kernel': {'gamma'}
}


# Define the parameter grids for each kernel function
classical_kernel_grid = [
    {'kernel_func': sigmoid_kernel, 'kernel_params': {'alpha': alpha, 'c': c}}
    for alpha, c in itertools.product([0.1, 0.5, 1.0], [0, 1, 5])
] + [
    {'kernel_func': laplacian_kernel, 'kernel_params': {'gamma': gamma}}
    for gamma in [0.1, 1.0, 10.0]
] + [
    {'kernel_func': anova_kernel, 'kernel_params': {'d': d, 'sigma': sigma}}
    for d, sigma in itertools.product([2, 3], [0.1, 1.0])
] + [
    {'kernel_func': chi_squared_kernel, 'kernel_params': {}}
] + [
    {'kernel_func': histogram_intersection_kernel, 'kernel_params': {}}
] + [
    {'kernel_func': linear_kernel, 'kernel_params': {}}
] + [
    {'kernel_func': polynomial_kernel, 'kernel_params': {'degree': degree, 'coef0': coef0}}
    for degree, coef0 in itertools.product([2, 3, 4], [0, 1])
] + [
    {'kernel_func': gaussian_kernel, 'kernel_params': {'gamma': gamma}}
    for gamma in [0.1, 1.0, 10.0]
] + [
    {'kernel_func': rbf_kernel, 'kernel_params': {'gamma': gamma}}
    for gamma in [0.1, 1.0, 10.0]
]

quantum_kernel_param_map = {
    'iqp_kernel': {'repeats'},
    'angle_embedding_kernel': {'rotation'},
    'amplitude_embedding_kernel': set()
}


quantum_kernel_grid = [
    {'kernel_func': iqp_kernel, 'kernel_params': {'repeats': repeats}}
    for repeats in [1, 2, 3]
] + [
    {'kernel_func': angle_embedding_kernel, 'kernel_params': {'rotation': rotation}}
    for rotation in ['X', 'Y', 'Z']
] + [
    {'kernel_func': amplitude_embedding_kernel, 'kernel_params': {}}
]

kernel_grid = classical_kernel_grid + quantum_kernel_grid
kernel_param_map = {
    **classical_kernel_param_map,
    **quantum_kernel_param_map
}


def select_kernels(kernel_grid, selected_kernels):
    """
    Filter the kernel grid to include only selected kernels.

    Parameters:
    - kernel_grid (list): The complete list of kernel configurations.
    - selected_kernels (list): The list of kernel function names to include.

    Returns:
    - filtered_grid (list): The filtered list of kernel configurations.
    """
    filtered_grid = [
        kernel for kernel in kernel_grid 
        if kernel['kernel_func'].__name__ in selected_kernels
    ]
    return filtered_grid
