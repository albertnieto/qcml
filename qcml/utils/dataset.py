import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_mnist_data():
    # Load MNIST dataset from sklearn
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Separate features and labels
    X = mnist.data
    y = mnist.target.astype(int)

    return X, y

def split_data(X, y, split_ratio):
    train_ratio, val_ratio = map(float, split_ratio.split('/'))
    train_ratio /= 100.0
    val_ratio /= 100.0
    if train_ratio + val_ratio != 1.0:
        raise ValueError("Split ratio must sum to 100 (e.g., '80/20').")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
    logger.debug(f"Data split into training samples: {len(y_train)}, validation samples: {len(y_val)}")
    return X_train, X_val, y_train, y_val