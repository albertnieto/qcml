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

# qcml/simulator/layers.py

from abc import ABC, abstractmethod
from typing import Any
from . import backend as bk
from . import gates


class QuantumGateLayer(ABC):
    @abstractmethod
    def forward(self, state: Any) -> Any:
        pass


class XLayer(QuantumGateLayer):
    def __init__(self, target: int, num_qubits: int):
        self.target = target
        self.num_qubits = num_qubits
        self.gate_matrix = gates.X_gate()

    def forward(self, state):
        # Expand gate matrix
        full_gate = self._expand_single_qubit_gate(self.gate_matrix, self.target)
        # Apply gate
        state = bk.dot(full_gate, state)
        return state

    def _expand_single_qubit_gate(self, gate_matrix, target_qubit):
        # Same as in QuantumSimulator
        operators = [gates.identity_gate()] * self.num_qubits
        operators[target_qubit] = gate_matrix
        full_gate = operators[0]
        for op in operators[1:]:
            full_gate = bk.kron(full_gate, op)
        return full_gate


# Similarly define other layers like YLayer, ZLayer, HLayer, etc.


# For parameterized gates like RX
class RXLayer(QuantumGateLayer):
    def __init__(self, target: int, theta, num_qubits: int):
        self.target = target
        self.theta = theta
        self.num_qubits = num_qubits
        self.gate_matrix = gates.RX_gate(theta)

    def forward(self, state):
        full_gate = self._expand_single_qubit_gate(self.gate_matrix, self.target)
        state = bk.dot(full_gate, state)
        return state

    def _expand_single_qubit_gate(self, gate_matrix, target_qubit):
        # Same as before
        operators = [gates.identity_gate()] * self.num_qubits
        operators[target_qubit] = gate_matrix
        full_gate = operators[0]
        for op in operators[1:]:
            full_gate = bk.kron(full_gate, op)
        return full_gate


# For two-qubit gates like CNOT
class CNOTLayer(QuantumGateLayer):
    def __init__(self, control: int, target: int, num_qubits: int):
        self.control = control
        self.target = target
        self.num_qubits = num_qubits
        self.gate_matrix = gates.CNOT_gate()

    def forward(self, state):
        full_gate = self._expand_two_qubit_gate(
            self.gate_matrix, self.control, self.target
        )
        state = bk.dot(full_gate, state)
        return state

    def _expand_two_qubit_gate(self, gate_matrix, control_qubit, target_qubit):
        # Build full gate matrix
        I = gates.identity_gate()
        zero_projector = bk.array([[1, 0], [0, 0]], dtype=bk.complex64)
        one_projector = bk.array([[0, 0], [0, 1]], dtype=bk.complex64)

        # Control on |0>
        gate_0 = [I] * self.num_qubits
        gate_0[control_qubit] = zero_projector
        gate_0_full = gate_0[0]
        for op in gate_0[1:]:
            gate_0_full = bk.kron(gate_0_full, op)

        # Control on |1>
        gate_1 = [I] * self.num_qubits
        gate_1[control_qubit] = one_projector
        gate_1[target_qubit] = gate_matrix
        gate_1_full = gate_1[0]
        for op in gate_1[1:]:
            gate_1_full = bk.kron(gate_1_full, op)

        full_gate = gate_0_full + gate_1_full
        return full_gate
