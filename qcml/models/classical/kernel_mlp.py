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

import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable, List, Dict, Union
from flax.training import train_state
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import time


class CenterKernelLayer(nn.Module):
    kernel_func: Callable
    kernel_params: Dict

    def __call__(self, x, rng):
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # Reshape to (batch_size, 1)

        batch_size, input_dim = x.shape

        centers_idx = jax.random.choice(rng, jnp.arange(x.shape[0]), shape=(batch_size,))
        centers = x[centers_idx]  # Shape: (batch_size, input_dim)

        def compute_kernel(x_i, centers_i):
            return self.kernel_func(x_i, centers_i, **self.kernel_params)

        kernel_output = jax.vmap(compute_kernel, in_axes=(0, 0))(x, centers)

        return kernel_output

# Define the KernelMappingLayer
class KernelMappingLayer(nn.Module):
    kernel_func: Callable
    kernel_params: Dict

    def __call__(self, x, w):
        # Calculate kernel value between input x and weight vector w for each sample in the batch
        kernel_output = jax.vmap(
            lambda x_i: self.kernel_func(x_i, w, **self.kernel_params),
            in_axes=(0)
        )(x)

        return kernel_output

# Define the main MLP module
class KernelMLPClassifierModule(nn.Module):
    hidden_layer_sizes: List[Union[int, str]]
    num_classes: int
    kernel_func: Callable
    kernel_params: Dict

    def setup(self):
        layers = []

        for layer_spec in self.hidden_layer_sizes:
            if isinstance(layer_spec, int):
                layers.append(nn.Dense(features=layer_spec))
            elif isinstance(layer_spec, str):
                if layer_spec.endswith('k'):
                    layers.append(CenterKernelLayer(
                        kernel_func=self.kernel_func,
                        kernel_params=self.kernel_params
                    ))
                elif layer_spec == 'k':
                    layers.append(KernelMappingLayer(
                        kernel_func=self.kernel_func,
                        kernel_params=self.kernel_params
                    ))
                else:
                    raise ValueError(f"Unknown layer specification {layer_spec}")
            else:
                raise ValueError(f"Unknown layer specification {layer_spec}")

        self.layers = layers
        self.output_layer = nn.Dense(features=self.num_classes)

    def __call__(self, x, rng=None):
        if rng is None:
            rng = self.make_rng('params')

        for i, layer in enumerate(self.layers):
            if isinstance(layer, CenterKernelLayer):
                x = layer(x, rng=rng)
            elif isinstance(layer, KernelMappingLayer):
                w = self.param(f'w_{i}', jax.nn.initializers.lecun_normal(), (x.shape[-1],))
                x = layer(x, w)
            else:
                x = layer(x)
                x = nn.relu(x)

        return self.output_layer(x)

# Define a pure JAX-based training class
class KernelMLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes: List[Union[int, str]],
        learning_rate_init: float,
        num_classes: int,
        kernel_func: Callable = None,
        kernel_params: Dict = None,
        max_iter: int = 3000,
        batch_size: int = 32,
        one_hot_encode: bool = True,
        early_stopping_rounds: int = 10,
        validation_fraction: float = 0.1,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.num_classes = num_classes
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

    def create_train_state(self, rng, input_shape):
        model = KernelMLPClassifierModule(
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.num_classes,
            kernel_func=self.kernel_func,
            kernel_params=self.kernel_params
        )
        params = model.init(rng, jnp.ones(input_shape))["params"]
        tx = optax.adam(self.learning_rate_init)
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        ), model

    @staticmethod
    @jax.jit
    def train_step(state, batch, rng):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["x"], rng=rng)
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits=logits, labels=batch["y"])
            )
            return loss

        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_value

    @staticmethod
    @jax.jit
    def eval_step(state, batch, rng):
        logits = state.apply_fn({"params": state.params}, batch["x"], rng=rng)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batch["y"]))
        return loss

    def fit(self, X_train, y_train):
        if self.one_hot_encode:
            y_train = jax.nn.one_hot(y_train, num_classes=self.num_classes)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.validation_fraction, random_state=42
        )

        input_shape = (self.batch_size, X_train.shape[-1])
        rng = jax.random.PRNGKey(0)
        state, model = self.create_train_state(rng, input_shape)

        best_loss = float('inf')
        best_state = None
        no_improvement_epochs = 0

        for epoch in range(1, self.max_iter + 1):
            start_time = time.time()
            epoch_loss = 0

            perm = jax.random.permutation(rng, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train_shuffled[i : i + self.batch_size]
                batch_y = y_train_shuffled[i : i + self.batch_size]

                if batch_X.shape[0] < self.batch_size:
                    padding_size = self.batch_size - batch_X.shape[0]
                    batch_X = jnp.pad(batch_X, ((0, padding_size), (0, 0)), mode='constant')
                    if self.one_hot_encode:
                        batch_y = jnp.pad(batch_y, ((0, padding_size), (0, 0)), mode='constant')
                    else:
                        batch_y = jnp.pad(batch_y, ((0, padding_size),), mode='constant')

                batch = {"x": batch_X, "y": batch_y}
                rng, batch_rng = jax.random.split(rng)
                state, loss_value = self.train_step(state, batch, batch_rng)
                epoch_loss += loss_value

            epoch_loss /= (len(X_train) // self.batch_size)
            epoch_time = time.time() - start_time

            val_loss = self.evaluate(state, model, X_val, y_val)
            print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = state
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= self.early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch}")
                    state = best_state
                    break

        self.state = state
        self.model = model
        return self

    def evaluate(self, state, model, X, y):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X), self.batch_size):
            batch_X = X[i : i + self.batch_size]
            batch_y = y[i : i + self.batch_size]

            if batch_X.shape[0] < self.batch_size:
                padding_size = self.batch_size - batch_X.shape[0]
                batch_X = jnp.pad(batch_X, ((0, padding_size), (0, 0)), mode='constant')
                if self.one_hot_encode:
                    batch_y = jnp.pad(batch_y, ((0, padding_size), (0, 0)), mode='constant')
                else:
                    batch_y = jnp.pad(batch_y, ((0, padding_size),), mode='constant')

            batch = {"x": batch_X, "y": batch_y}
            rng = jax.random.PRNGKey(1)
            loss = self.eval_step(state, batch, rng)
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches
        return average_loss

    def predict(self, X):
        num_samples = X.shape[0]  # Total number of samples in X
        logits_list = []
    
        print(f"Number of samples: {num_samples}")
        print(f"Batch size: {self.batch_size}")
    
        for i in range(0, num_samples, self.batch_size):
            batch_X = X[i:i + self.batch_size]
            actual_batch_size = batch_X.shape[0]
    
            print(f"Processing batch from {i} to {i + actual_batch_size} (actual batch size: {actual_batch_size})")
    
            if actual_batch_size < self.batch_size:
                # Pad the batch to the expected size if it's smaller than the batch size
                padding_size = self.batch_size - actual_batch_size
                batch_X = jnp.pad(batch_X, ((0, padding_size), (0, 0)), mode='constant')
                print(f"Padded batch size to {batch_X.shape}")
    
            # Predict logits for the batch
            rng = jax.random.PRNGKey(2)
            batch_logits = self.model.apply({"params": self.state.params}, batch_X, rng=rng)
    
            # Debugging: Print the shape of batch_logits
            print(f"batch_logits shape: {batch_logits.shape}")
    
            # Only keep the logits corresponding to the actual batch size (ignore padded parts)
            logits_list.extend(batch_logits[:actual_batch_size])
    
        # Concatenate all logits into a single array
        logits = jnp.array(logits_list)  # Use jnp.array to convert the list of arrays into a single array
        print(f"Total logits shape after concatenation: {logits.shape}")
    
        # Ensure we have the correct number of predictions
        if logits.shape[0] != num_samples:
            raise ValueError(f"Expected {num_samples} predictions, but got {logits.shape[0]}")
    
        return jnp.argmax(logits, axis=1)
