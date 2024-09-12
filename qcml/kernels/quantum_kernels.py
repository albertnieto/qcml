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

import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import logging
from qcml.utils.jax import min_max_scale

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)


def separable_kernel(
    x1,
    x2,
    encoding_layers=1,
    jit=False,
    dev_type="default.qubit.jax",
    qnode_kwargs={"interface": "jax", "diff_method": None},
):
    n_qubits = x1.shape[0]

    def construct_circuit():
        # Directly define the observable without re-checking for Hermiticity
        projector = np.array([[1.0, 0.0], [0.0, 0.0]])
        dev = qml.device(dev_type, wires=1)

        @qml.qnode(dev, **qnode_kwargs)
        def qubit_circuit(x):
            for layer in range(encoding_layers):
                qml.RX(-np.pi / 4, wires=0)
                qml.RY(-x[0], wires=0)
            for layer in range(encoding_layers):
                qml.RY(x[1], wires=0)
                qml.RX(np.pi / 4, wires=0)
            return qml.expval(qml.Hermitian(projector, wires=0))

        return qubit_circuit

    # Initialize the circuit
    qubit_circuit = construct_circuit()

    def full_circuit(x):
        probs = [
            qubit_circuit(jnp.array([x[i], x[i + n_qubits]])) for i in range(n_qubits)
        ]
        return jnp.prod(jnp.array(probs))

    if jit:
        full_circuit = jax.jit(full_circuit)

    # Combine x1 and x2 to form the input to the circuit
    z = jnp.concatenate((x1, x2))

    # Compute the kernel value
    kernel_value = full_circuit(z)

    return kernel_value

def projected_quantum_kernel(
    x1,
    x2,
    embedding="Hamiltonian",
    t=1.0 / 3,
    trotter_steps=5,
    gamma_factor=1.0,
    jit=False,
    dev_type="default.qubit.jax",
    qnode_kwargs={"interface": "jax-jit", "diff_method": None},
):
    n_features = x1.shape[0]

    # Determine the number of qubits based on the embedding type
    if embedding == "IQP":
        n_qubits = n_features
        def embedding_fn(x):
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=2)

    elif embedding == "Hamiltonian":
        n_qubits = n_features + 1
        rotation_angles = jnp.array(np.random.uniform(size=(n_qubits, 3)) * np.pi * 2)
        evol_time = t / trotter_steps * (n_qubits - 1)
        def embedding_fn(x):
            for i in range(n_qubits):
                qml.Rot(
                    rotation_angles[i, 0],
                    rotation_angles[i, 1],
                    rotation_angles[i, 2],
                    wires=i,
                )
            for __ in range(trotter_steps):
                for j in range(n_qubits - 1):
                    qml.IsingXX(x[j] * evol_time, wires=[j, j + 1])
                    qml.IsingYY(x[j] * evol_time, wires=[j, j + 1])
                    qml.IsingZZ(x[j] * evol_time, wires=[j, j + 1])

    # Initialize the quantum device
    dev = qml.device(dev_type, wires=n_qubits)

    # Define the quantum circuit within a QNode
    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x):
        embedding_fn(x)
        return (
            [qml.expval(qml.PauliX(wires=i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        )

    # Optionally JIT compile the circuit
    if jit:
        circuit = jax.jit(circuit)

    # Compute expectation values for both input vectors
    expvals_x1 = jnp.array(circuit(x1))
    expvals_x2 = jnp.array(circuit(x2))

    # Compute the squared differences
    diff = expvals_x1 - expvals_x2
    diff_squared_sum = jnp.sum(diff**2)

    # Calculate default gamma value based on the variance of the expectation values
    default_gamma = 1 / jnp.var(expvals_x1) / n_features

    # Compute the final kernel value using the Gaussian kernel formula
    kernel_value = jnp.exp(-default_gamma * gamma_factor * diff_squared_sum)

    return kernel_value


def iqp_embedding_kernel(
    x1,
    x2,
    repeats=1,
    jit=False,
    dev_type="default.qubit.jax",
    qnode_kwargs={"interface": "jax-jit", "diff_method": None},
):
    n_qubits = x1.shape[0]  # Number of qubits equals the number of features
    dev = qml.device(dev_type, wires=n_qubits)
    # logger.debug(f"IQP embedding kernel is using {n_qubits} qubits")

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x1, x2):
        qml.IQPEmbedding(x1, wires=range(n_qubits), n_repeats=repeats)
        qml.adjoint(qml.IQPEmbedding(x2, wires=range(n_qubits), n_repeats=repeats))
        return qml.expval(
            qml.PauliZ(0)
        )  # Expectation value of the Pauli-Z operator on the first qubit

    if jit:
        circuit = jax.jit(circuit)

    kernel_value = circuit(x1, x2)
    
    # Add a small jitter to prevent identical values
    kernel_value += 1e-8 * jnp.sign(kernel_value)

    return kernel_value


def angle_embedding_kernel(
    x1,
    x2,
    rotation="Y",
    jit=False,
    dev_type="default.qubit.jax",
    qnode_kwargs={"interface": "jax-jit", "diff_method": None},
):
    n_qubits = x1.shape[0]
    dev = qml.device(dev_type, wires=n_qubits)
    # logger.debug(f"Angle embedding kernel is using {n_qubits} qubits")

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation=rotation)
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation=rotation)
        return qml.expval(qml.PauliZ(0))

    if jit:
        circuit = jax.jit(circuit)

    return circuit(x1, x2)


def amplitude_embedding_kernel(
    x1,
    x2,
    pad_with=0.0,
    normalize=True,  # This controls the normalization within the embedding
    jit=False,
    dev_type="default.qubit.jax",
    qnode_kwargs={"interface": "jax-jit", "diff_method": None},
):
    n_qubits = int(np.ceil(np.log2(max(len(x1), len(x2)))))
    dev = qml.device(dev_type, wires=n_qubits)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x1, x2, n_qubits):
        logger.info(f"Amplitude circuit has {n_qubits} qubits")
        qml.AmplitudeEmbedding(x1, wires=range(n_qubits), pad_with=pad_with, normalize=normalize)
        qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(n_qubits), pad_with=pad_with, normalize=normalize)
        return qml.expval(qml.PauliZ(0))

    if jit:
        circuit = jax.jit(circuit)

    return circuit(x1, x2, n_qubits)
