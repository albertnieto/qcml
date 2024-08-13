import pennylane as qml
import jax
import jax.numpy as jnp
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import logging
import time

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)

# Helper function for batched evaluation
def chunk_vmapped_fn(fn, start=0, max_vmap=None):
    def _batched_fn(x):
        batch_size = x.shape[0]
        if max_vmap is None or batch_size <= max_vmap:
            return fn(x)
        else:
            result = []
            for i in range(start, batch_size, max_vmap):
                result.append(fn(x[i : i + max_vmap]))
            return jnp.concatenate(result)
    return _batched_fn

# Common function to compute kernel matrix
def kernel_matrix(circuit, X, Y=None, max_vmap=250):
    if Y is None:
        Y = X
    Z = jnp.array(
        [jnp.concatenate((X[i], Y[j])) for i in range(len(X)) for j in range(len(Y))]
    )
    batched_circuit = chunk_vmapped_fn(jax.vmap(circuit, 0), max_vmap=max_vmap)
    kernel_values = batched_circuit(Z)[:, 0]
    return np.array(kernel_values).reshape((len(X), len(Y)))

# IQP Kernel
def iqp_kernel(X, Y=None, repeats=1, jit=True, dev_type="default.qubit.jax", qnode_kwargs={"interface": "jax-jit", "diff_method": None}, max_vmap=250):
    n_qubits = X.shape[1]
    dev = qml.device(dev_type, wires=n_qubits)
    logger.debug(f"IQP embedding kernel is using {n_qubits} qubits")
    
    scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    X = scaler.fit_transform(X)
    X = jnp.array(X)
    if Y is not None:
        Y = scaler.transform(Y)
        Y = jnp.array(Y)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x):
        qml.IQPEmbedding(
            x[:n_qubits], wires=range(n_qubits), n_repeats=repeats
        )
        qml.adjoint(
            qml.IQPEmbedding(
                x[n_qubits:], wires=range(n_qubits), n_repeats=repeats,
            )
        )
        return qml.probs()

    if jit:
        circuit = jax.jit(circuit)

    return kernel_matrix(circuit, X, Y, max_vmap)

# Angle Embedding Kernel
def angle_embedding_kernel(X, Y=None, rotation='Y', jit=True, dev_type="default.qubit.jax", qnode_kwargs={"interface": "jax-jit", "diff_method": None}, max_vmap=250):
    n_qubits = X.shape[1]
    dev = qml.device(dev_type, wires=n_qubits)
    logger.debug(f"Angle embedding kernel is using {n_qubits} qubits")

    scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    X = scaler.fit_transform(X)
    X = jnp.array(X)
    if Y is not None:
        Y = scaler.transform(Y)
        Y = jnp.array(Y)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x):
        qml.AngleEmbedding(x[:n_qubits], wires=range(n_qubits), rotation=rotation)
        qml.adjoint(
            qml.AngleEmbedding(x[n_qubits:], wires=range(n_qubits), rotation=rotation)
        )
        return qml.probs()

    if jit:
        circuit = jax.jit(circuit)

    return kernel_matrix(circuit, X, Y, max_vmap)

# Amplitude Embedding Kernel
def amplitude_embedding_kernel(X, Y=None, jit=True, dev_type="default.qubit.jax", qnode_kwargs={"interface": "jax-jit", "diff_method": None}, max_vmap=250):
    n_features = X.shape[1]
    n_qubits = n_features // 2
    if n_features % 2 != 0:
        raise ValueError("Number of features must be even for amplitude embedding.")

    logger.debug(f"Amplitude embedding kernel is using {n_qubits} qubits")

    dev = qml.device(dev_type, wires=n_qubits)

    X = jnp.array(X)
    if Y is not None:
        Y = jnp.array(Y)

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(x):
        qml.AmplitudeEmbedding(x[:n_qubits], wires=range(n_qubits), normalize=True, pad_with=True)
        qml.adjoint(
            qml.AmplitudeEmbedding(x[n_qubits:], wires=range(n_qubits), normalize=True, pad_with=True)
        )
        return qml.probs()

    if jit:
        circuit = jax.jit(circuit)

    return kernel_matrix(circuit, X, Y, max_vmap)
