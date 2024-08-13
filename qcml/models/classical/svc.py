from sklearn.svm import SVC as SVC_base
from qic.kernels import (
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
import numpy as np


class SVCCustom(SVC_base):
    def __init__(
            self,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            max_iter=-1,
            random_state=None,
            **kernel_params,
    ):
        # Define a mapping from string to kernel function
        kernel_map = {
            "sigmoid": sigmoid_kernel,
            "laplacian": laplacian_kernel,
            "anova": anova_kernel,
            "chi_squared": chi_squared_kernel,
            "histogram_intersection": histogram_intersection_kernel,
            "linear": linear_kernel,
            "polynomial": polynomial_kernel,
            "gaussian": gaussian_kernel,
            "rbf": rbf_kernel,
        }

        # Check if the kernel string is in the kernel map
        if kernel in kernel_map:
            kernel_function = kernel_map[kernel]
            def custom_kernel(X, Y):
                return np.array([[kernel_function(x, y, **kernel_params) for y in Y] for x in X])
            kernel = custom_kernel
        elif kernel != "precomputed":
            raise ValueError(f"Unknown kernel '{kernel}' specified. Available kernels are: {list(kernel_map.keys())} or 'precomputed'.")

        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )