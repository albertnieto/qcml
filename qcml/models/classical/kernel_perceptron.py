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
import jax
from jax import jit
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin


class KernelPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        kernel="precomputed",
        n_iter=-1,
        random_state=42,
        params=None,
        max_iter=1000,
    ):
        """
        Kernel Perceptron classifier.

        Parameters:
        kernel (str or callable): The kernel function to use. If "precomputed", a precomputed kernel matrix is expected.
        n_iter (int): Number of iterations for training. If -1, the algorithm will run until convergence or max_iter.
        random_state (int): Seed for random number generator.
        params (dict): Parameters to pass to the kernel function.
        max_iter (int): Maximum number of iterations to run if n_iter is -1.
        """
        self.kernel = kernel
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.params = params if params is not None else {}
        self.random_state = random_state
        self._rng = jax.random.PRNGKey(random_state)
        self.fitted = False
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.classes_ = jnp.array([0.0, 1.0], dtype=jnp.int32)

    def _is_fitted(self):
        """Check if the model is fitted."""
        return self.fitted

    def _compute_kernel_matrix(self, X):
        """Compute the kernel matrix for the input data X."""
        kernel_func = lambda x_, y_: self.kernel(x_, y_, **self.params)
        K = jax.vmap(lambda x: jax.vmap(lambda y: kernel_func(x, y))(X))(X)
        return K

    @staticmethod
    @jit
    def _fit_iteration(K, y, alpha):
        """
        Perform a single iteration of the perceptron training.

        Parameters:
        K (array): Kernel matrix.
        y (array): Labels.
        alpha (array): Alpha values.

        Returns:
        array: Updated alpha values.
        """

        def body_fun(i, alpha):
            condition = jnp.sign(jnp.sum(K[:, i] * alpha * y)) != y[i]
            return jax.lax.cond(
                condition, lambda a: a.at[i].add(1.0), lambda a: a, alpha
            )

        alpha = jax.lax.fori_loop(0, K.shape[0], body_fun, alpha)
        return alpha

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        X (array): Training data.
        y (array): Training labels.
        """
        if not callable(self.kernel) and self.kernel != "precomputed":
            raise ValueError("No valid kernel function provided.")

        if self.kernel == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed kernel must be a square matrix")
            K = X
        else:
            K = self._compute_kernel_matrix(X)

        alpha = jnp.zeros(K.shape[0])
        converged = False
        iter_idx = 0

        if self.n_iter != -1:
            for _ in range(self.n_iter):
                alpha = self._fit_iteration(K, y, alpha)
        else:
            while not converged and iter_idx < self.max_iter:
                alpha_new = self._fit_iteration(K, y, alpha)
                converged = jnp.all(alpha_new == alpha)
                alpha = alpha_new
                iter_idx += 1

        self.alpha = alpha

        sv = self.alpha > 1e-5
        self.alpha = self.alpha[sv]
        self.sv = X[sv] if self.kernel != "precomputed" else jnp.arange(K.shape[0])[sv]
        self.sv_y = y[sv]
        self.fitted = True

    def project(self, X):
        """
        Project the input data using the fitted model.

        Parameters:
        X (array): Input data.

        Returns:
        array: Projected values.
        """
        if self.kernel == "precomputed":
            y_predict = jax.vmap(
                lambda i: jnp.sum(self.alpha * self.sv_y * X[i, self.sv])
            )(jnp.arange(len(X)))
        elif callable(self.kernel):
            kernel_func = lambda x_, y_: self.kernel(x_, y_, **self.params)
            y_predict = jax.vmap(
                lambda x: jnp.sum(
                    self.alpha
                    * self.sv_y
                    * jax.vmap(lambda sv: kernel_func(x, sv))(self.sv)
                )
            )(X)
        else:
            raise ValueError("No valid kernel function provided.")
        return y_predict

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        X (array): Input data.

        Returns:
        array: Predicted class labels.
        """
        X = jnp.atleast_2d(X)
        return jnp.sign(self.project(X))

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        X (array): Input data.

        Returns:
        array: Predicted class probabilities.
        """
        decision_function = self.project(X)
        proba_class_1 = 1 / (1 + jnp.exp(-decision_function))
        proba_class_0 = 1 - proba_class_1
        y_pred_proba = jnp.vstack((proba_class_0, proba_class_1)).T
        return y_pred_proba

    def score(self, X, y):
        """
        Calculate the accuracy of the model.

        Parameters:
        X (array): Test data.
        y (array): True labels for the test data.

        Returns:
        float: Accuracy of the model.
        """
        y_pred = self.predict(X)
        accuracy = jnp.mean(y_pred == y)
        return accuracy
