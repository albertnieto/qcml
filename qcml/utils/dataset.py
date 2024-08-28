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

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_mnist_data():
    # Load MNIST dataset from sklearn
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    # Separate features and labels
    X = mnist.data
    y = mnist.target.astype(int)

    return X, y


def split_data(X, y, split_ratio):
    train_ratio, val_ratio = map(float, split_ratio.split("/"))
    train_ratio /= 100.0
    val_ratio /= 100.0
    if train_ratio + val_ratio != 1.0:
        raise ValueError("Split ratio must sum to 100 (e.g., '80/20').")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42
    )
    logger.debug(
        f"Data split into training samples: {len(y_train)}, validation samples: {len(y_val)}"
    )
    return X_train, X_val, y_train, y_val


def validate_input_data(
    X=None,
    y=None,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    split_ratio="80/20",
):
    if (X is not None and y is None) or (X is None and y is not None):
        raise ValueError("Both X and y must be provided if using whole dataset.")

    if (X_train is not None and y_train is None) or (
        X_train is None and y_train is not None
    ):
        raise ValueError(
            "Both X_train and y_train must be provided if using pre-split dataset."
        )

    if (X_val is not None and y_val is None) or (X_val is None and y_val is not None):
        raise ValueError(
            "Both X_val and y_val must be provided if using pre-split dataset."
        )

    if X is not None and split_ratio is None:
        raise ValueError("Split ratio must be provided if using whole dataset.")

    if X is not None and y is not None:
        X_train, X_val, y_train, y_val = split_data(X, y, split_ratio)
        logger.info(
            f"Data split into training and validation sets using split ratio: {split_ratio}"
        )
    elif X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError(
            "Insufficient data provided. Provide either whole dataset or pre-split dataset."
        )

    return X_train, X_val, y_train, y_val
