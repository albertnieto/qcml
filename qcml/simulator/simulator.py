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

from typing import Dict, Any
from . import backend as bk
from . import gates

class QuantumSimulator:
    """
    Simulates the execution of a quantum circuit.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = bk.zeros((2 ** num_qubits,), dtype=bk.complex64)
        self.state = self.state.at[0].set(1.0 + 0.0j)  # Initialize to |0...0>
        self.global_key = bk.PRNGKey(42) if bk.backend_name == 'jax' else None
    
    def _expand_single_qubit_gate(self, gate_matrix, target_qubit):
        # Create identity operators for other qubits
        operators = [gates.identity_gate()] * self.num_qubits
        operators[target_qubit] = gate_matrix
        # Compute the tensor product
        full_gate = operators[0]
        for op in operators[1:]:
            full_gate = bk.kron(full_gate, op)
        return full_gate
    
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
    
    @bk.jit
    def _apply_gate(self, gate_info: Dict[str, Any]):
        name = gate_info['name']
        targets = gate_info['targets']
        params = gate_info.get('params', {})
    
        if name in ['RX', 'RY', 'RZ']:
            theta = params.get('theta', 0)
            gate_matrix = getattr(gates, f"{name}_gate")(theta)
        else:
            gate_matrix = getattr(gates, f"{name}_gate")()
    
        if len(targets) == 1:
            # Single-qubit gate
            full_gate = self._expand_single_qubit_gate(gate_matrix, targets[0])
            self.state = bk.dot(full_gate, self.state)
        elif len(targets) == 2:
            # Two-qubit gate
            full_gate = self._expand_two_qubit_gate(gate_matrix, targets[0], targets[1])
            self.state = bk.dot(full_gate, self.state)
        else:
            raise ValueError("Only single and two-qubit gates are supported.")
    
    def run(self, circuit):
        for gate_info in circuit.gates:
            self._apply_gate(gate_info)
    
    def measure_all(self, num_shots=1):
        probabilities = bk.abs(self.state) ** 2
        outcomes = bk.array(range(2 ** self.num_qubits))
        if bk.backend_name == 'jax':
            self.global_key, subkey = bk.split(self.global_key)
            samples = bk.choice(subkey, outcomes, shape=(num_shots,), p=probabilities)
        elif bk.backend_name == 'torch':
            samples = bk.choice(probabilities, num_samples=num_shots, replacement=True)
        else:
            raise NotImplementedError("Measurement not implemented for this backend.")
        return samples
    
    def get_probabilities(self):
        return bk.abs(self.state) ** 2
