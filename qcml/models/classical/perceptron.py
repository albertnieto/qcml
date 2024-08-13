import os
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax.lib import xla_bridge

from jax import jit, grad
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=5, random_state=42, max_iter=1000):
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.random_state = random_state
        self._rng = jax.random.PRNGKey(random_state)
        self.fitted = False
        self.weights = None
        self.bias = None
        self.classes_ = jnp.array([0.0, 1.0], dtype=jnp.int32)

    def _is_fitted(self):
        return self.fitted

    @staticmethod
    @jit
    def _fit_iteration(X, y, n_iter, weights, bias):
        def body_fun(i, params):
            w, b = params
            condition = y[i] * (jnp.dot(X[i], w) + b) <= 0
            new_w = jax.lax.cond(condition, lambda w: w + y[i] * X[i], lambda w: w, w)
            new_b = jax.lax.cond(condition, lambda b: b + y[i], lambda b: b, b)
            return new_w, new_b

        def iter_body(_, params):
            return jax.lax.fori_loop(0, X.shape[0], body_fun, params)

        weights, bias = jax.lax.fori_loop(0, n_iter, iter_body, (weights, bias))
        return weights, bias

    def fit(self, X, y):
        y = jnp.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for perceptron
        self.weights = jnp.zeros(X.shape[1])
        self.bias = 0.0

        self.weights, self.bias = self._fit_iteration(X, y, self.n_iter, self.weights, self.bias)
        self.fitted = True

    def project(self, X):
        return jnp.dot(X, self.weights) + self.bias

    def predict(self, X):
        X = jnp.atleast_2d(X)
        return jnp.sign(self.project(X))

    def predict_proba(self, X):
        decision_function = self.project(X)
        proba_class_1 = 1 / (1 + jnp.exp(-decision_function))
        proba_class_0 = 1 - proba_class_1
        y_pred_proba = jnp.vstack((proba_class_0, proba_class_1)).T
        return y_pred_proba

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = jnp.mean(y_pred == y)
        return accuracy
