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

import jax.numpy as jnp
import jax
import numpy as np
import logging
import time
import concurrent.futures
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def kernel_transform(
    X_train,
    X_val,
    y_train,
    y_val,
    kernel_func,
    kernel_params=None,
    n_centers=10,
    scale_data=None,
):
    """
    Apply a kernel transformation to the training and validation data.

    Parameters:
    - X_train: Training data, shape (n_train_samples, n_features).
    - X_val: Validation data, shape (n_val_samples, n_features).
    - kernel_func: Kernel function to be applied.
    - kernel_params: Additional parameters for the kernel function.
    - n_centers: Number of centers to choose from X_train.
    - scale_data: 'standardization' or 'minmax' to choose the scaling method, or None for no scaling.

    Returns:
    - X_train_transformed: Transformed training data, shape (n_train_samples, n_centers).
    - X_val_transformed: Transformed validation data, shape (n_val_samples, n_centers).
    """
    # Optionally scale the data
    if scale_data == "standardization":
        scaler = StandardScaler()
    elif scale_data == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    # Choose random centers from X_train
    rng = np.random.default_rng()
    indices = rng.choice(X_train.shape[0], size=n_centers, replace=False)
    centers = X_train[indices]

    # Ensure kernel_params is a dictionary
    if kernel_params is None:
        kernel_params = {}

    # Compute the kernel matrix for X_train and centers
    X_train_transformed = compute_gram_matrix(
        X_train, centers, kernel_func, **kernel_params
    )

    # Compute the kernel matrix for X_val and centers
    X_val_transformed = compute_gram_matrix(
        X_val, centers, kernel_func, **kernel_params
    )

    return X_train_transformed, X_val_transformed, y_train, y_val


def jitted_gram_matrix_batched(X1, X2, kernel_func, batch_size=50, **kernel_params):
    start_time = time.time()

    n_samples_X1 = X1.shape[0]
    n_samples_X2 = X2.shape[0]
    Gram_matrix = jnp.zeros((n_samples_X1, n_samples_X2))

    def wrapped_kernel_func(x1, x2):
        return kernel_func(x1, x2, **kernel_params)

    kernel_func_vmap = jax.jit(
        jax.vmap(jax.vmap(wrapped_kernel_func, in_axes=(None, 0)), in_axes=(0, None))
    )

    # Trigger JIT compilation by calling the function with dummy data
    _ = kernel_func_vmap(X1[:1], X2[:1])
    compilation_time = time.time() - start_time

    device_info = jax.devices()[0]

    logger.debug(
        f"JIT and vmap applied for Gram matrix computation on device: {device_info.platform.upper()}. "
        f"Compilation time: {compilation_time:.4f}s. "
        f"X1 shape: {X1.shape}, X2 shape: {X2.shape}. "
        f"Kernel: {kernel_func.__name__}, JIT enabled: {jax.config.read('jax_enable_x64')}. "
        f"Expected output shape: ({X1.shape[0]}, {X2.shape[0]})"
    )

    for i in range(0, n_samples_X1, batch_size):
        for j in range(0, n_samples_X2, batch_size):
            X1_batch = X1[i : i + batch_size]
            X2_batch = X2[j : j + batch_size]

            batch_result = kernel_func_vmap(X1_batch, X2_batch)
            jax.block_until_ready(
                batch_result
            )  # Ensures computation is done and memory is cleared
            Gram_matrix = Gram_matrix.at[i : i + batch_size, j : j + batch_size].set(
                batch_result
            )

    execution_time = time.time() - start_time

    logger.debug(
        f"Execution time for batched Gram matrix computation: {execution_time:.4f}s. "
        f"Real output shape: {Gram_matrix.shape}"
    )

    return Gram_matrix


def jitted_gram_matrix(X1, X2, kernel_func, **kernel_params):
    start_time = time.time()

    # Create a wrapper to pass kernel_params to kernel_func
    def wrapped_kernel_func(x1, x2):
        return kernel_func(x1, x2, **kernel_params)

    # Vectorized and JIT-compiled Gram matrix computation
    kernel_func_vmap = jax.jit(
        jax.vmap(jax.vmap(wrapped_kernel_func, in_axes=(None, 0)), in_axes=(0, None))
    )

    # Trigger JIT compilation by calling the function with dummy data
    _ = kernel_func_vmap(X1[:1], X2[:1])
    compilation_time = time.time() - start_time

    device_info = jax.devices()[0]

    logger.debug(
        f"JIT and vmap applied for Gram matrix computation on device: {device_info.platform.upper()}. "
        f"Compilation time: {compilation_time:.4f}s. "
        f"X1 shape: {X1.shape}, X2 shape: {X2.shape}. "
        f"Kernel: {kernel_func.__name__}, JIT enabled: {jax.config.read('jax_enable_x64')}. "
        f"Expected output shape: ({X1.shape[0]}, {X2.shape[0]})"
    )

    # Measure execution time of the actual Gram matrix computation
    execution_start_time = time.time()
    gram_matrix = kernel_func_vmap(X1, X2)
    execution_time = time.time() - execution_start_time

    logger.debug(
        f"Execution time for Gram matrix computation: {execution_time:.4f}s. "
        f"Real output shape: {gram_matrix.shape}"
    )

    return gram_matrix


def compute_kernel_parallel(X1, X2, kernel_func, **kernel_params):
    n_samples_X1 = X1.shape[0]
    n_samples_X2 = X2.shape[0]
    Gram_matrix = jnp.zeros((n_samples_X1, n_samples_X2))

    def compute_entry(i, j):
        return kernel_func(X1[i], X2[j], **kernel_params)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(compute_entry, i, j): (i, j)
            for i in range(n_samples_X1)
            for j in range(n_samples_X2)
        }
        logger.debug(
            f"Computing Gram matrix for kernel function: {kernel_func.__name__}, kernel_params: {kernel_params}, input sizes: {X1.shape}, {X2.shape}, split in {len(futures)}"
        )
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            Gram_matrix = Gram_matrix.at[i, j].set(future.result())

    return Gram_matrix


def batched_gram_matrix(X1, X2, kernel_func, batch_size=100, **kernel_params):
    n_samples_X1 = X1.shape[0]
    n_samples_X2 = X2.shape[0]
    Gram_matrix = jnp.zeros((n_samples_X1, n_samples_X2))

    for i in range(0, n_samples_X1, batch_size):
        for j in range(0, n_samples_X2, batch_size):
            X1_batch = X1[i : i + batch_size]
            X2_batch = X2[j : j + batch_size]
            batch_result = jnp.array(
                [
                    [kernel_func(x1, x2, **kernel_params) for x2 in X2_batch]
                    for x1 in X1_batch
                ]
            )
            Gram_matrix = Gram_matrix.at[i : i + batch_size, j : j + batch_size].set(
                batch_result
            )

    return Gram_matrix


def compute_gram_matrix(X1, X2, kernel_func, **kernel_params):
    logger.debug(
        f"Computing Gram matrix for kernel function: {kernel_func.__name__}, kernel_params: {kernel_params}, input sizes: {X1.shape}, {X2.shape}"
    )
    Gram_matrix = batched_gram_matrix(
        X1, X2, kernel_func, batch_size=100, **kernel_params
    )
    return jnp.array(Gram_matrix)
