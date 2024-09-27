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

from .jax.chi_squared_kernel import chi_squared_kernel as chi_squared_kernel_jax
from .numpy.chi_squared_kernel import chi_squared_kernel as chi_squared_kernel_numpy
from .torch.chi_squared_kernel import chi_squared_kernel as chi_squared_kernel_torch


def chi_squared_kernel(x1, x2, backend="numpy"):
    """
    Compute the Chi-Squared kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.
        backend (str): Backend to use ('jax', 'numpy', 'torch'). Default is 'numpy'.

    Returns:
        Kernel result based on the selected backend.
    """
    if backend == "jax":
        return chi_squared_kernel_jax(x1, x2)
    elif backend == "torch":
        return chi_squared_kernel_torch(x1, x2)
    else:
        return chi_squared_kernel_numpy(x1, x2)
