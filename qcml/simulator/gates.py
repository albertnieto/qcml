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

from . import backend as bk

def X_gate():
    return bk.array([[0, 1],
                     [1, 0]], dtype=bk.complex64)

def Y_gate():
    return bk.array([[0, -1j],
                     [1j, 0]], dtype=bk.complex64)

def Z_gate():
    return bk.array([[1, 0],
                     [0, -1]], dtype=bk.complex64)

def H_gate():
    return (1 / bk.sqrt(2)) * bk.array([[1, 1],
                                        [1, -1]], dtype=bk.complex64)

def S_gate():
    return bk.array([[1, 0],
                     [0, 1j]], dtype=bk.complex64)

def T_gate():
    return bk.array([[1, 0],
                     [0, bk.exp(1j * bk.pi / 4)]], dtype=bk.complex64)

def RX_gate(theta):
    return bk.array([[bk.cos(theta / 2), -1j * bk.sin(theta / 2)],
                     [-1j * bk.sin(theta / 2), bk.cos(theta / 2)]], dtype=bk.complex64)

def RY_gate(theta):
    return bk.array([[bk.cos(theta / 2), -bk.sin(theta / 2)],
                     [bk.sin(theta / 2), bk.cos(theta / 2)]], dtype=bk.complex64)

def RZ_gate(theta):
    return bk.array([[bk.exp(-1j * theta / 2), 0],
                     [0, bk.exp(1j * theta / 2)]], dtype=bk.complex64)

def CNOT_gate():
    return bk.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=bk.complex64)

def identity_gate():
    return bk.eye(2, dtype=bk.complex64)
