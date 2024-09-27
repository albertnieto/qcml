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

import pytest
import os
from qcml.simulator.backend import set_backend

def test_set_backend_jax():
    set_backend('jax')
    import qcml.simulator.backend as bk
    assert bk.backend_name == 'jax'
    assert bk.array is not None  # Ensure array function is set

def test_set_backend_torch():
    set_backend('torch')
    import qcml.simulator.backend as bk
    assert bk.backend_name == 'torch'
    assert bk.array is not None  # Ensure array function is set

def test_invalid_backend():
    with pytest.raises(ImportError):
        set_backend('invalid_backend')
