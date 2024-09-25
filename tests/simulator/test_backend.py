import pytest
import os
from qcml.simulator.backend import set_backend

def test_set_backend_jax():
    set_backend('jax')
    import qcml.simulator.backend as bk
    assert bk.backend_name == 'jax'

def test_set_backend_torch():
    set_backend('torch')
    import qcml.simulator.backend as bk
    assert bk.backend_name == 'torch'

def test_invalid_backend():
    with pytest.raises(ImportError):
        set_backend('invalid_backend')
