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

from .jax.anova_kernel import anova_kernel as anova_kernel_jax
from .numpy.anova_kernel import anova_kernel as anova_kernel_numpy
from .torch.anova_kernel import anova_kernel as anova_kernel_torch


def anova_kernel(x1, x2, backend="numpy", d=2, sigma=1):
    """
    Compute the ANOVA kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.
        d (int): Degree of the kernel.
        sigma (float): Scaling parameter.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return anova_kernel_jax(x1, x2, d, sigma)
    elif backend == "torch":
        return anova_kernel_torch(x1, x2, d, sigma)
    else:
        return anova_kernel_numpy(x1, x2, d, sigma)
