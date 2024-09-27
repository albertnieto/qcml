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

# qcml/simulator/layer_circuit.py
from typing import List, Dict, Any, Optional
from . import backend as bk
from .layers import QuantumGateLayer


class LayerBasedCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.layers: List[QuantumGateLayer] = []

    def add_layer(self, layer: QuantumGateLayer):
        self.layers.append(layer)

    def get_gate_sequence(self) -> List[Dict[str, Any]]:
        # Convert layers to gate_info dictionaries if needed
        gate_sequence = []
        for layer in self.layers:
            # Assuming each layer can provide its gate_info
            gate_info = layer.get_gate_info()
            gate_sequence.append(gate_info)
        return gate_sequence

    def run(self, initial_state=None):
        if initial_state is None:
            state = bk.zeros((2**self.num_qubits,), dtype=bk.complex64)
            if bk.backend_name == "jax":
                state = state.at[0].set(1.0 + 0.0j)  # Initialize to |0...0>
            else:
                state[0] = 1.0 + 0.0j
        else:
            state = initial_state
        for layer in self.layers:
            state = layer.forward(state)
        return state
