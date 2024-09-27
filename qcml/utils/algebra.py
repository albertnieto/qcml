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


def is_matrix_normal(matrix):
    return np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix)


def is_matrix_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)


def is_matrix_unitary(matrix):
    return np.allclose(np.linalg.inv(matrix), matrix.conj().T)
