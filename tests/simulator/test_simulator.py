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
from qcml.simulator.simulator import QuantumSimulator
from qcml.simulator.circuit import QuantumCircuit
from qcml.simulator.backend import set_backend

@pytest.fixture(autouse=True)
def setup_backend():
    set_backend('jax')

def test_init_simulator():
    sim = QuantumSimulator(2)
    assert sim.num_qubits == 2
    assert sim.state.shape == (4,)
    assert sim.state[0] == 1.0 + 0.0j

#def test_apply_gate():
#    sim = QuantumSimulator(2)
#    qc = QuantumCircuit(2)
#    qc.x(0)
#    sim.run(qc)
    
    # Use .item() to extract the scalar value regardless of backend (JAX or PyTorch)
#    assert abs(sim.state[1].item()) != 0.0

def test_measure_all():
    sim = QuantumSimulator(2)
    result = sim.measure_all(num_shots=1000)
    assert result.shape == (1000,)
