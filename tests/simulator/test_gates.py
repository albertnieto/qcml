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
from qcml.simulator.gates import X_gate, Y_gate, Z_gate
from qcml.simulator.backend import set_backend

@pytest.fixture(autouse=True)
def setup_backend():
    set_backend('jax')

def test_x_gate():
    x = X_gate()
    assert x.shape == (2, 2)
    assert x[0, 1] == 1
    assert x[1, 0] == 1

def test_y_gate():
    y = Y_gate()
    assert y.shape == (2, 2)
    assert y[0, 1] == -1j
    assert y[1, 0] == 1j

def test_z_gate():
    z = Z_gate()
    assert z.shape == (2, 2)
    assert z[0, 0] == 1
    assert z[1, 1] == -1
