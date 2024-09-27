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

# qcml/simulator/graph_circuit.py

import networkx as nx

class GraphCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.graph = nx.DiGraph()
        self.node_counter = 0  # To assign unique IDs to nodes

    def add_gate(self, gate_name: str, targets: List[int], params: Optional[Dict[str, Any]] = None):
        node_id = self.node_counter
        self.graph.add_node(node_id, gate_name=gate_name, targets=targets, params=params)
        if node_id > 0:
            self.graph.add_edge(node_id - 1, node_id)
        self.node_counter += 1

    def get_gate_sequence(self) -> List[Dict[str, Any]]:
        # Topologically sort the graph to get execution order
        sorted_nodes = list(nx.topological_sort(self.graph))
        gate_sequence = []
        for node_id in sorted_nodes:
            node_data = self.graph.nodes[node_id]
            gate_info = {
                "name": node_data["gate_name"],
                "targets": node_data["targets"],
                "params": node_data.get("params", {}),
            }
            gate_sequence.append(gate_info)
        return gate_sequence
