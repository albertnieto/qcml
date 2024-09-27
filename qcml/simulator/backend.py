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

import importlib
import os, sys

_backend = os.environ.get("SIMULATOR_BACKEND", "jax")

if _backend == "jax":
    import jax
    import jax.numpy as np
    from jax import jit, random

    array = np.array
    zeros = np.zeros
    eye = np.eye
    kron = np.kron
    dot = np.dot
    exp = np.exp
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    conj = np.conj
    abs = np.abs
    complex64 = np.complex64
    float32 = np.float32
    int32 = np.int32
    backend_name = "jax"
    PRNGKey = random.PRNGKey
    choice = random.choice
    split = random.split
    jit = jit
elif _backend == "torch":
    import torch
    import torch as np
    from torch import complex64, float32, int32
    from torch import eye, zeros, conj
    from torch import sin, cos, exp, sqrt
    from torch import abs as torch_abs

    array = torch.tensor
    kron = torch.kron
    dot = torch.matmul
    abs = lambda x: torch_abs(x)
    backend_name = "torch"
    # For randomness, we can use torch's random functions
    PRNGKey = None  # Not needed for torch
    choice = torch.multinomial
    split = None  # Not needed for torch
    jit = lambda f: f  # No-op decorator
else:
    raise ImportError(f"Unsupported backend '{_backend}'")


def set_backend(backend_name: str):
    """
    Set the computational backend: 'jax' or 'torch'.
    """
    global _backend
    _backend = backend_name
    os.environ["SIMULATOR_BACKEND"] = _backend
    
    # Re-import the current module to refresh backend
    importlib.reload(sys.modules[__name__])