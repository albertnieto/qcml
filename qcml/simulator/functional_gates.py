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

# qcml/simulator/functional_gates.py

from . import backend as bk
from . import gates

def x_gate(state, target, num_qubits):
    # Expand gate
    gate_matrix = gates.X_gate()
    full_gate = _expand_single_qubit_gate(gate_matrix, target, num_qubits)
    state = bk.dot(full_gate, state)
    return state

def rx_gate(state, target, theta, num_qubits):
    gate_matrix = gates.RX_gate(theta)
    full_gate = _expand_single_qubit_gate(gate_matrix, target, num_qubits)
    state = bk.dot(full_gate, state)
    return state

def _expand_single_qubit_gate(gate_matrix, target_qubit, num_qubits):
    operators = [gates.identity_gate()] * num_qubits
    operators[target_qubit] = gate_matrix
    full_gate = operators[0]
    for op in operators[1:]:
        full_gate = bk.kron(full_gate, op)
    return full_gate