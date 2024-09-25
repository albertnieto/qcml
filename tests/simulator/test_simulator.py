from qcml.simulator.simulator import QuantumSimulator
from qcml.simulator.circuit import QuantumCircuit

def test_init_simulator():
    sim = QuantumSimulator(2)
    assert sim.num_qubits == 2
    assert sim.state.shape == (4,)
    assert sim.state[0] == 1.0 + 0.0j

def test_apply_gate():
    sim = QuantumSimulator(2)
    qc = QuantumCircuit(2)
    qc.x(0)
    sim.run(qc)
    # Assert that after applying the X gate, the state is updated
    assert sim.state[1] != 0.0

def test_measure_all():
    sim = QuantumSimulator(2)
    result = sim.measure_all(num_shots=1000)
    assert result.shape == (1000,)
