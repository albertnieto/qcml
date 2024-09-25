from qcml.simulator.gates import X_gate, Y_gate, Z_gate, H_gate, S_gate, T_gate

def test_x_gate():
    x = X_gate()
    assert x.shape == (2, 2)
    assert x[0, 1] == 1
    assert x[1, 0] == 1

def test_y_gate():
    y = Y_gate()
    assert y.shape == (2, 2)
    assert y[0, 1] == -1j
    assert y[1, 0] == 1j

def test_z_gate():
    z = Z_gate()
    assert z.shape == (2, 2)
    assert z[0, 0] == 1
    assert z[1, 1] == -1
