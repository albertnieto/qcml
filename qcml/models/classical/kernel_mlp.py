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
from flax.training import train_state
from sklearn.base import BaseEstimator, ClassifierMixin


class KernelLayer(nn.Module):
    n_centers: int
    kernel_func: callable
    kernel_params: dict

    def setup(self):
        self.centers = self.param(
            "centers", nn.initializers.uniform(), (self.n_centers, None)
        )

    def __call__(self, x):
        if self.centers.shape[-1] is None:
            self.centers = jax.random.uniform(
                self.make_rng("params"), (self.n_centers, x.shape[-1])
            )
        x_expanded = jnp.expand_dims(x, 1)
        centers_expanded = jnp.expand_dims(self.centers, 0)
        return self.kernel_func(x_expanded, centers_expanded, **self.kernel_params)


class KernelMLPClassifier(nn.Module):
    layer_config: list
    num_classes: int

    def setup(self):
        self.layers = tuple(
            nn.Dense(**params) if layer_type == "dense" else KernelLayer(**params)
            for layer_type, params in self.layer_config
        )
        self.output_layer = nn.Dense(features=self.num_classes)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return self.output_layer(x)


class KernelMLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        layer_config,
        learning_rate,
        num_classes,
        num_epochs=3000,
        batch_size=32,
        one_hot_encode=True,
    ):
        self.layer_config = layer_config
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        self.model = None
        self.state = None

    def create_train_state(self, rng, input_shape):
        params = self.model.init(rng, jnp.ones(input_shape))["params"]
        tx = optax.adam(self.learning_rate)
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    def fit(self, X_train, y_train):
        if self.one_hot_encode:
            y_train = jax.nn.one_hot(y_train, num_classes=self.num_classes)

        input_shape = (1, X_train.shape[-1])
        self.model = KernelMLPClassifier(
            layer_config=self.layer_config, num_classes=self.num_classes
        )
        rng = jax.random.PRNGKey(0)
        self.state = self.create_train_state(rng, input_shape)

        for epoch in range(self.num_epochs):
            for i in range(0, len(X_train), self.batch_size):
                batch = {
                    "x": X_train[i : i + self.batch_size],
                    "y": y_train[i : i + self.batch_size],
                }
                self.state = self.train_step(self.state, batch)
        return self

    def train_step(self, state, batch):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["x"])
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits=logits, labels=batch["y"])
            )
            return loss

        # JIT-compile only the loss function
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    @jax.jit
    def predict(self, X):
        logits = self.model.apply({"params": self.state.params}, X)
        return jnp.argmax(logits, axis=1)

    def eval_step(self, state, batch):
        logits = state.apply_fn({"params": state.params}, batch["x"])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batch["y"]))
        return loss, logits
