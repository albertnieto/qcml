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

import gc
from numba import cuda
import logging

logger = logging.getLogger(__name__)


# WARNING: destroys context
def clear_gpu_memory():
    device = cuda.get_current_device()
    device.reset()
    gc.collect()
    logger.info("GPU memory cleared.")
