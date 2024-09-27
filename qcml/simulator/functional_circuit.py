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

# qcml/simulator/functional_circuit.py


def functional_circuit(state, num_qubits):
    from .functional_gates import x_gate, rx_gate

    state = x_gate(state, target=0, num_qubits=num_qubits)
    state = rx_gate(state, target=1, theta=0.5, num_qubits=num_qubits)
    # Add more gate function calls as needed
    return state
