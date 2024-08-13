# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing models to be used in benchmarks."""

from qic.models.quantum.circuit_centric import CircuitCentricClassifier
from qic.models.quantum.convolutional_neural_network import (
    ConvolutionalNeuralNetwork,
)
from qic.models.quantum.data_reuploading import (
    DataReuploadingClassifier,
    DataReuploadingClassifierNoScaling,
    DataReuploadingClassifierNoCost,
    DataReuploadingClassifierNoTrainableEmbedding,
    DataReuploadingClassifierSeparable,
)
from qic.models.quantum.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierOnlyNN,
    DressedQuantumCircuitClassifierSeparable,
)

from qic.models.quantum.iqp_kernel import IQPKernelClassifier
from qic.models.quantum.iqp_variational import IQPVariationalClassifier
from qic.models.quantum.projected_quantum_kernel import ProjectedQuantumKernel
from qic.models.quantum.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable
)
from qic.models.quantum.quantum_kitchen_sinks import QuantumKitchenSinks
from qic.models.quantum.quantum_metric_learning import QuantumMetricLearner
from qic.models.quantum.quanvolutional_neural_network import QuanvolutionalNeuralNetwork
from qic.models.quantum.separable import (
    SeparableVariationalClassifier,
    SeparableKernelClassifier,
)
from qic.models.quantum.tree_tensor import TreeTensorClassifier
from qic.models.quantum.vanilla_qnn import VanillaQNN
from qic.models.quantum.weinet import WeiNet

from qic.models.classical.svc import SVCCustom as SVCCustom_Base
from qic.models.classical.perceptron import Perceptron as PerceptronCustom_base
from qic.models.classical.kernel_perceptron import KernelPerceptron as KernelPerceptron_base
from qic.models.classical.kernel_mlp import KernelMLPClassifier

from qic.models.quantum.ansatz_embedding_kernel import AnsatzEmbeddingKernel

from sklearn.svm import SVC as SVC_base
from sklearn.linear_model import Perceptron as Perceptron_base
from sklearn.neural_network import MLPClassifier as MLP

__all__ = [
    "CircuitCentricClassifier",
    "ConvolutionalNeuralNetwork",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierNoScaling",
    "DataReuploadingClassifierNoCost",
    "DataReuploadingClassifierNoTrainableEmbedding",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierOnlyNN",
    "DressedQuantumCircuitClassifierSeparable",
    "IQPKernelClassifier",
    "IQPVariationalClassifier",
    "ProjectedQuantumKernel",
    "QuantumBoltzmannMachine",
    "QuantumBoltzmannMachineSeparable",
    "QuantumKitchenSinks",
    "QuantumMetricLearner",
    "QuanvolutionalNeuralNetwork",
    "SeparableVariationalClassifier",
    "SeparableKernelClassifier",
    "TreeTensorClassifier",
    "VanillaQNN",
    "WeiNet",
    "MLPClassifier",
    "SVC",
    "SVCCustom",
    "Perceptron",
    "KernelPerceptron",
    "KernelMLPClassifier",
    "AnsatzEmbeddingKernel",
]

class MLPClassifier(MLP):
    def __init__(
            self,
            hidden_layer_sizes=(100, 100),  # The ith element represents the number of neurons in the ith hidden layer
            activation="relu",              # Activation function for the hidden layer ('relu', 'logistic', 'tanh', 'identity')
            solver="adam",                  # The solver for weight optimization ('lbfgs', 'sgd', 'adam')
            alpha=0.0001,                   # L2 penalty (regularization term) parameter
            batch_size="auto",              # Size of minibatches for stochastic optimizers
            learning_rate="constant",       # Learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive')
            learning_rate_init=0.001,       # Initial learning rate used
            power_t=0.5,                    # The exponent for inverse scaling learning rate
            max_iter=3000,                  # Maximum number of iterations
            shuffle=True,                   # Whether to shuffle samples in each iteration
            random_state=None,              # Seed for the random number generator
            tol=1e-4,                       # Tolerance for the optimization
            verbose=False,                  # Whether to print progress messages to stdout
            warm_start=False,               # Whether to reuse the solution of the previous call to fit
            momentum=0.9,                   # Momentum for gradient descent update
            nesterovs_momentum=True,        # Whether to use Nesterov’s momentum
            early_stopping=False,           # Whether to use early stopping to terminate training when validation score is not improving
            validation_fraction=0.1,        # The proportion of training data to set aside as validation set for early stopping
            beta_1=0.9,                     # Exponential decay rate for estimates of first moment vector in Adam
            beta_2=0.999,                   # Exponential decay rate for estimates of second moment vector in Adam
            epsilon=1e-8,                   # Value for numerical stability in Adam
            n_iter_no_change=10,            # Number of iterations with no improvement to wait before early stopping
            max_fun=15000,                  # Maximum number of function calls in the optimizer
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

class SVC(SVC_base):
    def __init__(
            self,
            C=1.0,               # Regularization parameter. The strength of the regularization is inversely proportional to C
            kernel="precomputed",
            degree=3,            # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels
            gamma="scale",       # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            coef0=0.0,           # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'
            shrinking=True,      # Whether to use the shrinking heuristic
            probability=False,   # Whether to enable probability estimates
            tol=0.001,           # Tolerance for stopping criterion
            max_iter=-1,         # Hard limit on iterations within solver, or -1 for no limit
            random_state=None,   # Seed for the random number generator
    ):
        super().__init__(
            C=C,
            kernel=kernel,        # Specifies the kernel type to be used in the algorithm ('rbf' by default)
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
        
class Perceptron(Perceptron_base):
    def __init__(
            self,
            penalty=None,           # No penalty (no regularization)
            alpha=0.0001,           # Regularization strength
            l1_ratio=0.15,          # Ratio of L1 regularization (used only if penalty is 'elasticnet')
            fit_intercept=True,     # Whether to calculate the intercept for this model
            max_iter=1000,          # Maximum number of passes over the training data
            tol=0.001,              # The stopping criterion (tolerance)
            shuffle=True,           # Whether to shuffle the training data after each epoch
            verbose=0,              # The verbosity level
            eta0=1.0,               # Constant by which the updates are multiplied
            n_jobs=None,            # Number of CPU cores used during the cross-validation loop
            random_state=0,         # Seed for the random number generator
            early_stopping=False,   # Whether to use early stopping to terminate training
            validation_fraction=0.1,# The proportion of training data to set aside as validation set
            n_iter_no_change=5,     # Number of iterations with no improvement to wait before early stopping
            class_weight=None,      # Weights associated with classes
            warm_start=False        # Whether to reuse the solution of the previous call to fit
    ):
        super().__init__(
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            eta0=eta0,
            n_jobs=n_jobs,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
        )

class KernelPerceptron(KernelPerceptron_base):
    def __init__(
            self,
            kernel="precomputed",   # The kernel type to be used in the algorithm (default is "precomputed")
            n_iter=200,             # Number of passes over the training data
            random_state=42,        # Seed for the random number generator
            params=None,            # Additional parameters for the kernel function
            max_iter=1000,          # Maximum number of passes over the training data
    ):
        super().__init__(
            kernel=kernel,
            n_iter=n_iter,
            random_state=random_state,
            params=params,
            max_iter=max_iter,
        )

# Extend SVCCustom
class SVCCustom(SVCCustom_Base):
    def __init__(
            self,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            max_iter=-1,
            random_state=None,
            **kernel_params,
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            **kernel_params,
        )
        
class PerceptronCustom(PerceptronCustom_base):
    def __init__(
            self,
            n_iter=5,
            random_state=42,
            max_iter=1000
    ):
        super().__init__(
            n_iter=n_iter,
            random_state=random_state,
            max_iter=max_iter
        )