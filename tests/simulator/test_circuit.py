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

from qcml.simulator.circuit import QuantumCircuit


def test_create_circuit():
    qc = QuantumCircuit(num_qubits=3)
    assert qc.num_qubits == 3
    assert qc.gates == []


def test_add_gate():
    qc = QuantumCircuit(num_qubits=3)
    qc.x(0)
    assert qc.gates == [{"name": "X", "targets": [0]}]
    qc.h(1)
    assert qc.gates[-1] == {"name": "H", "targets": [1]}


def test_custom_gate():
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate("CUSTOM", [0, 1], {"theta": 0.5})
    assert qc.gates == [{"name": "CUSTOM", "targets": [0, 1], "params": {"theta": 0.5}}]
