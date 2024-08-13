import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def display_image_and_label_torch(dataloader):
    """
    Display an image and its corresponding label from a PyTorch DataLoader.

    Parameters:
    dataloader (torch.utils.data.DataLoader): The DataLoader to fetch the data from.

    Returns:
    None
    """
    # Get a batch of training data
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    # Get the first image from the batch
    img = train_features[0]
    
    # Check if the image has a single channel (grayscale) or three channels (RGB)
    if img.size(0) == 1:
        img = img.squeeze()  # Remove single-channel dimension for grayscale image
        plt.imshow(img, cmap="gray")
    elif img.size(0) == 3:
        # For RGB image, permute the dimensions to move channels to the last dimension
        img = img.permute(1, 2, 0)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported number of channels. Expected 1 or 3 channels.")
    
    plt.axis('off')  # Hide axis
    plt.show()
    
    print(f"Label: {train_labels[0]}")

def plot_samples_eurosat(X, y, n_samples=10):
    """
    Plots n_samples of the images and their corresponding labels from the EuroSAT dataset.

    Parameters:
    X (np.ndarray): The array of flattened images.
    y (np.ndarray): The array of labels.
    n_samples (int): The number of samples to plot.
    """
    # Ensure we don't plot more samples than we have
    n_samples = min(n_samples, len(X))
    
    # Determine the image shape
    n_features = X.shape[1]
    side_length = int(np.sqrt(n_features))
    
    # Verify if the side_length is an integer
    if side_length * side_length != n_features:
        raise ValueError("The number of features does not form a perfect square, please check the data.")
    
    image_shape = (side_length, side_length)
    
    plt.figure(figsize=(15, 15))
    
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        img = X[i].reshape(image_shape)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    
    plt.show()
    
def plot_samples_mnist_pca(X, y, classes, num_samples=4, shape=[1,1]):
    """
    Plot a sample of images for each class specified in 'classes'.

    Parameters:
    - X: numpy array, features data (e.g., images after PCA transformation)
    - y: numpy array, labels corresponding to each image
    - classes: list of integers, classes to plot samples from
    - num_samples: int, number of samples to plot for each class

    Returns:
    - None
    """
    # Initialize plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.set(style='whitegrid')

    # Plot samples for each class
    for i, digit in enumerate(classes):
        # Find indices of samples belonging to current class
        indices = (y == digit)
        X_digit = X[indices]

        # Check if X_digit is not empty
        if len(X_digit) == 0:
            print(f"No samples found for class {digit}. Skipping plotting.")
            continue

        # Plot the first num_samples samples
        for j in range(min(num_samples, len(X_digit))):  # Ensure we don't exceed available samples
            # Reshape X_digit[j] to the appropriate shape (adjust to your actual PCA-determined shape)
            reshaped_image = X_digit[j].reshape((shape[0], shape[1]))  # Example shape, replace with actual shape

            plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            plt.imshow(reshaped_image, cmap='gray')  # Plot the reshaped image
            plt.title(f'Class {digit}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_samples_mnist(X, y, classes, num_samples=4):
    """
    Plot a sample of images for each class specified in 'classes'.

    Parameters:
    - X: numpy array, features data (e.g., images)
    - y: numpy array, labels corresponding to each image
    - classes: list of integers, classes to plot samples from
    - num_samples: int, number of samples to plot for each class

    Returns:
    - None
    """
    # Initialize plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.set(style='whitegrid')

    # Plot samples for each class
    for i, digit in enumerate(classes):
        # Find indices of samples belonging to current class
        indices = np.where(y == digit)[0]
        if len(indices) == 0:
            print(f"No samples found for class {digit}. Skipping plotting.")
            continue

        # Plot the first num_samples samples
        for j in range(min(num_samples, len(indices))):  # Ensure we don't exceed available samples
            plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            plt.imshow(X[indices[j]].reshape((28, 28)), cmap='gray')  # Plot the image
            plt.title(f'Class {digit}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_classifier_decision_boundary(clf, data_points, labels, title=None):
    """
    Visualizes the decision boundary of a classifier with a decision function in 2D space.

    Args:
        clf (object): Trained classifier with a decision_function method.
        data_points (numpy.ndarray): Array of data points used in the visualization.
        labels (numpy.ndarray): Array of labels corresponding to the data points.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    # Generate a grid of points for visualization
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)

    # Compute the decision function for each point in the grid
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x_test = np.array([x1_range[i], x2_range[j]])
            Z[i, j] = clf.decision_function([x_test])

    # Create a custom colormap
    cmap = ListedColormap(['blue', 'red'])
            
    # Plotting
    plt.figure(figsize=(5, 3))
    plt.contourf(X1, X2, Z, levels=20, cmap=cmap, alpha=0.6)

    # Plot the data points
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, marker='o', s=50, cmap='bwr', edgecolor='k')

    plt.xlabel('X1')
    plt.ylabel('X2')
    
    # Set the title if provided
    if title:
        plt.title(title)

    plt.show()


def plot_classifier_decision_boundary_3d(clf, data_points, labels, title=None):
    """
    Visualizes the decision boundary of a classifier with a decision function in 3D space.

    Args:
        clf (object): Trained classifier with a decision_function method.
        data_points (numpy.ndarray): Array of data points used in the visualization.
        labels (numpy.ndarray): Array of labels corresponding to the data points.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    # Generate a grid of points for visualization
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)

    # Compute the decision function for each point in the grid
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x_test = np.array([x1_range[i], x2_range[j]])
            Z[i, j] = clf.decision_function([x_test])

    # Create a custom colormap
    cmap = ListedColormap(['blue', 'red'])
            
    # Plotting
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap=cmap, alpha=0.6)

    # Plot the data points
    ax.scatter(data_points[:, 0], data_points[:, 1], np.zeros_like(data_points[:, 0]), c=labels, marker='o', s=20, cmap='bwr')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Function Value')
    
    # Set the title if provided
    if title:
        plt.title(title)

    plt.show()
