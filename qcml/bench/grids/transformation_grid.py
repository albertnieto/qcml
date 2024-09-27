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

from qcml.bench.grids.kernel_grid import kernel_grid
from qcml.utils.kernel import kernel_transform
import logging

logger = logging.getLogger(__name__)


def get_kernel_transform():

    # Common parameters for transformations
    n_centers_options = [10, 20, 50, 100]
    scale_data_options = ["standardization", "minmax"]

    # Transformation parameter grids (renamed to transformation_params)
    transformation_params = [
        {},  # For None (no transformation)
    ]

    for kernel_entry in kernel_grid:
        kernel_func = kernel_entry["kernel_func"]
        kernel_params = kernel_entry["kernel_params"]

        # Create a transformation grid for the current kernel
        transformation_entry = {
            "kernel_func": [kernel_func],
            "kernel_params": [kernel_params],
            "n_centers": n_centers_options,
            "scale_data": scale_data_options,
        }

        # Add the entry to the transformation_params list
        transformation_params.append(transformation_entry)

    # Generate transformation_func
    transformation_func = [None] + [kernel_transform] * (len(transformation_params) - 1)
    logger.debug(
        f"Getting kernel transformation with {len(transformation_func)} combinations"
    )

    return transformation_func, transformation_params
