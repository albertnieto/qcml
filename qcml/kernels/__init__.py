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


from .classical.rbf_kernel import rbf_kernel
from .classical.sigmoid_kernel import sigmoid_kernel
from .classical.laplacian_kernel import laplacian_kernel
from .classical.anova_kernel import anova_kernel
from .classical.chi_squared_kernel import chi_squared_kernel
from .classical.histogram_intersection_kernel import histogram_intersection_kernel
from .classical.linear_kernel import linear_kernel
from .classical.polynomial_kernel import polynomial_kernel
from .classical.gaussian_kernel import gaussian_kernel
from .quantum_kernels import (
    separable_kernel,
    projected_quantum_kernel,
    iqp_embedding_kernel,
    angle_embedding_kernel,
    amplitude_embedding_kernel,
)

__all__ = [
    "sigmoid_kernel",
    "laplacian_kernel",
    "anova_kernel",
    "chi_squared_kernel",
    "histogram_intersection_kernel",
    "linear_kernel",
    "polynomial_kernel",
    "gaussian_kernel",
    "rbf_kernel",
    "separable_kernel",
    "projected_quantum_kernel",
    "iqp_embedding_kernel",
    "angle_embedding_kernel",
    "amplitude_embedding_kernel",
]
