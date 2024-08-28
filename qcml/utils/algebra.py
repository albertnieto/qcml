import numpy as np


def is_matrix_normal(matrix):
    return np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix)


def is_matrix_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)


def is_matrix_unitary(matrix):
    return np.allclose(np.linalg.inv(matrix), matrix.conj().T)
