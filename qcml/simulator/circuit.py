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

from typing import List, Dict, Any, Optional
from . import gates


class QuantumCircuit:
    """
    Represents a quantum circuit.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: List[Dict[str, Any]] = []

    # Methods to add gates
    def x(self, target: int):
        self.gates.append({"name": "X", "targets": [target]})

    def y(self, target: int):
        self.gates.append({"name": "Y", "targets": [target]})

    def z(self, target: int):
        self.gates.append({"name": "Z", "targets": [target]})

    def h(self, target: int):
        self.gates.append({"name": "H", "targets": [target]})

    def s(self, target: int):
        self.gates.append({"name": "S", "targets": [target]})

    def t(self, target: int):
        self.gates.append({"name": "T", "targets": [target]})

    def rx(self, target: int, theta):
        self.gates.append(
            {"name": "RX", "targets": [target], "params": {"theta": theta}}
        )

    def ry(self, target: int, theta):
        self.gates.append(
            {"name": "RY", "targets": [target], "params": {"theta": theta}}
        )

    def rz(self, target: int, theta):
        self.gates.append(
            {"name": "RZ", "targets": [target], "params": {"theta": theta}}
        )

    def cnot(self, control: int, target: int):
        self.gates.append({"name": "CNOT", "targets": [control, target]})

    def add_gate(
        self, name: str, targets: List[int], params: Optional[Dict[str, Any]] = None
    ):
        self.gates.append({"name": name, "targets": targets, "params": params})

    def get_gate_sequence(self) -> List[Dict[str, Any]]:
        return self.gates
