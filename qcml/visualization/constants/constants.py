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

import sympy as sp
from constants.fractions import common_fractions

# Define symbolic constants
hbar, pi = sp.symbols("ħ π", real=True, positive=True)
zero, one = sp.symbols("0 1", real=True)

# Pauli matrices
sigma_x, sigma_y, sigma_z = sp.symbols("σ_x σ_y σ_z", commutative=False)

# Hadamard gate
H = 1 / sp.sqrt(2) * sp.Matrix([[1, 1], [1, -1]])

# Phase gate
S = sp.Matrix([[1, 0], [0, sp.I]])

# CNOT gate
CNOT = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# Bloch sphere coordinates
theta, phi = sp.symbols("θ φ", real=True)
theta_half = theta / 2
phi_half = phi / 2

# Bloch sphere basis states
ket_0 = sp.Matrix([[1], [0]])
ket_1 = sp.Matrix([[0], [1]])

# Bell states
bell_00 = 1 / sp.sqrt(2) * (ket_0 * ket_0.T + ket_1 * ket_1.T)
bell_01 = 1 / sp.sqrt(2) * (ket_0 * ket_0.T - ket_1 * ket_1.T)
bell_10 = 1 / sp.sqrt(2) * (ket_0 * ket_1.T + ket_1 * ket_0.T)
bell_11 = 1 / sp.sqrt(2) * (ket_0 * ket_1.T - ket_1 * ket_0.T)

# Entangled states
psi_minus = 1 / sp.sqrt(2) * (ket_0 * ket_0.T - ket_1 * ket_1.T)

# Quantum gates
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])

# Dirac notation
ket_psi = sp.symbols("|ψ⟩")

# Bell measurement
BM = sp.Matrix(
    [
        [1, 0, 0, 0],
        [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2), 0],
        [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)

# Controlled-phase gate
CPHASE = sp.Matrix(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sp.exp(sp.I * pi / 2)]]
)

# Create a dictionary of symbolic constants
quantum_constants = {
    "hbar": hbar,
    "pi": pi,
    "zero": zero,
    "one": one,
    "sigma_x": sigma_x,
    "sigma_y": sigma_y,
    "sigma_z": sigma_z,
    "H": H,
    "S": S,
    "CNOT": CNOT,
    "theta": theta,
    "phi": phi,
    "ket_0": ket_0,
    "ket_1": ket_1,
    "bell_00": bell_00,
    "bell_01": bell_01,
    "bell_10": bell_10,
    "bell_11": bell_11,
    "psi_minus": psi_minus,
    "X": X,
    "Y": Y,
    "Z": Z,
    "ket_psi": ket_psi,
    "BM": BM,
    "CPHASE": CPHASE,
    **common_fractions,
}


def is_symbolic_constant(amplitude, tolerance=1e-10):
    """
    Check if an amplitude corresponds to a symbolic constant within a specified tolerance.

    Parameters:
    - amplitude (float): The amplitude to check.
    - tolerance (float): Tolerance for comparing amplitudes with symbolic constants.
                        The default value is 1e-10.

    Returns:
    str or None: The symbolic representation if a match is found, otherwise None.
    """
    return next(
        (
            symbol
            for constant, symbol in quantum_constants.items()
            if np.isclose(amplitude, constant, atol=tolerance)
        ),
        None,
    )
