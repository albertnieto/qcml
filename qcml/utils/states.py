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
