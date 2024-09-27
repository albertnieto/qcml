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

import numpy as np


def zero_state_array(n):
    """
    Create an n-qubit zero state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: Zero state for n qubits.
    """
    return np.kron(*[np.array([1, 0]) for _ in range(n)])


def one_state_array(n):
    """
    Create an n-qubit one state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: One state for n qubits.
    """
    return np.kron(*[np.array([0, 1]) for _ in range(n)])


def plus_state_array(n):
    """
    Create an n-qubit plus state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: Plus state for n qubits.
    """
    return np.kron(*[np.array([1, 1]) / np.sqrt(2) for _ in range(n)])


def minus_state_array(n):
    """
    Create an n-qubit minus state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: Minus state for n qubits.
    """
    return np.kron(*[np.array([1, -1]) / np.sqrt(2) for _ in range(n)])


def i_plus_state_array(n):
    """
    Create an n-qubit |i+⟩ state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: |i+⟩ state for n qubits.
    """
    return np.kron(
        *[(np.array([1, 0]) + 1j * np.array([0, 1])) / np.sqrt(2) for _ in range(n)]
    )


def i_minus_state_array(n):
    """
    Create an n-qubit |i-⟩ state.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - np.ndarray: |i-⟩ state for n qubits.
    """
    return np.kron(
        *[(np.array([1, 0]) - 1j * np.array([0, 1])) / np.sqrt(2) for _ in range(n)]
    )


def bell_state_1_array():
    """
    Create the first Bell state |Φ⁺⟩.

    Returns:
    - np.ndarray: |Φ⁺⟩ Bell state.
    """
    return (
        np.kron(np.array([1, 0]), np.array([1, 0]))
        + np.kron(np.array([0, 1]), np.array([0, 1]))
    ) / np.sqrt(2)


def bell_state_2_array():
    """
    Create the second Bell state |Φ⁻⟩.

    Returns:
    - np.ndarray: |Φ⁻⟩ Bell state.
    """
    return (
        np.kron(np.array([1, 0]), np.array([1, 0]))
        - np.kron(np.array([0, 1]), np.array([0, 1]))
    ) / np.sqrt(2)


def bell_state_3_array():
    """
    Create the third Bell state |Ψ⁺⟩.

    Returns:
    - np.ndarray: |Ψ⁺⟩ Bell state.
    """
    return (
        np.kron(np.array([1, 0]), np.array([0, 1]))
        + np.kron(np.array([0, 1]), np.array([1, 0]))
    ) / np.sqrt(2)


def bell_state_4_array():
    """
    Create the fourth Bell state |Ψ⁻⟩.

    Returns:
    - np.ndarray: |Ψ⁻⟩ Bell state.
    """
    return (
        np.kron(np.array([1, 0]), np.array([0, 1]))
        - np.kron(np.array([0, 1]), np.array([1, 0]))
    ) / np.sqrt(2)
