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

import torch


def rbf_kernel(x1, x2, gamma=None):
    """
    Compute the RBF kernel using PyTorch.

    Args:
        x1 (torch.Tensor): First vector with shape (n_features,).
        x2 (torch.Tensor): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        torch.Tensor: Kernel value.
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[0]
    sq_dist = torch.sum((x1[:, None] - x2[None, :]) ** 2, dim=-1)
    return torch.exp(-gamma * sq_dist)
