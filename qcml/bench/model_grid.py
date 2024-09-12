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

model_grid = {
    "LinearSVC": {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
        "kernel": ["precomputed"],
    },
    "KernelPerceptron": {
        "n_iter": [-1],  # Number of iterations
        "max_iter": [2000],  # Maximum number of iterations
        "kernel": ["precomputed"],
    },
    "Perceptron": {
        "eta0": [0.01, 0.1, 0.5, 1.0],  # Learning rate
    },
    "PerceptronCustom": {
        "n_iter": [1000],  # Number of iterations
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [
            (50,),
            (100,),
            (100, 50),
            (100, 100),
        ],  # Configurations for hidden layers
        "learning_rate_init": [
            0.001,
            0.01,
        ],  # Initial learning rate (specific to MLPClassifier)
        "max_iter": [3000],  # Maximum number of iterations
    },
    "MLPClassifierCustom": {
        "hidden_layer_sizes": [
            (50,),
            (100,),
            (100, 50),
            (100, 100),
        ],  # Configurations for hidden layers
        "learning_rate": [
            0.001,
            0.01,
        ],  # Learning rate for optimizer (specific to MLPClassifierCustom)
        "max_iter": [2000],  # Maximum number of iterations
        "num_classes": [2],  # Number of output classes
    },
    "KernelMLPClassifier": {
        "hidden_layer_sizes": [
            ("k", 50),
            ("k", 100),
            ("k", 100, 50),
            ("k", 100, 100),
        ],  # Configurations for hidden layers
        "learning_rate_init": [
            0.001,
            0.01,
        ],  # Initial learning rate (specific to MLPClassifier)
        "max_iter": [2000],  # Maximum number of iterations
        "num_classes": [2],  # Number of output classes
    },
    "QuanvolutionalNeuralNetwork": {
        "max_vmap": [32],
        "batch_size": [32],
        "learning_rate": [0.0001, 0.001, 0.01],
        "n_qchannels": [1, 5, 10],
        "qkernel_shape": [2, 3],
        "kernel_shape": [2, 3, 5],
    },
}
