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

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler


class EuroSAT(Dataset):
    """
    EuroSAT dataset class for PyTorch.

    Parameters:
    -----------
    root : str
        Root directory of the dataset.
    train : bool, optional
        If True, creates dataset from training set, otherwise from test set (default is True).
    transform : callable, optional
        A function/transform that takes in an image and returns a transformed version.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.

    Attributes:
    -----------
    img_labels : pandas.DataFrame
        DataFrame containing image paths and labels.
    img_dir : str
        Directory where images are stored.
    index_to_label : dict
        Mapping from index to label.
    height : int
        Height of the images.
    width : int
        Width of the images.
    data : list
        List containing image data.
    targets : numpy.ndarray
        Array containing the labels.
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, replace=False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.replace = replace

        # Load annotations
        self.annotations_file = os.path.join(
            self.root, "train.csv" if self.train else "test.csv"
        )
        self.img_labels = pd.read_csv(self.annotations_file)

        # Load label map
        self.img_dir = self.root
        self.index_to_label = self._load_label_map(
            os.path.join(self.root, "label_map.json")
        )

        # Get image size
        self.height, self.width = self._get_size()

        # Load images and targets
        self.data = self._load_images()
        self.targets = self.img_labels.iloc[:, 2].to_numpy()

    def _load_label_map(self, label_map_path):
        """
        Load the label map from a JSON file.

        Parameters:
        -----------
        label_map_path : str
            Path to the label map JSON file.

        Returns:
        --------
        dict
            Dictionary mapping from index to label.
        """
        with open(label_map_path, "r") as f:
            return json.load(f)

    def _get_size(self):
        """
        Get the size of the images.

        Returns:
        --------
        tuple
            A tuple containing the height and width of the images.
        """
        img_name = str(self.img_labels.iloc[0, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        return image.shape[-2], image.shape[-1]

    def _load_images(self):
        """
        Load all images from the dataset.

        Returns:
        --------
        list
            List of image tensors.
        """
        images = []
        for img_name in self.img_labels.iloc[:, 1]:
            img_path = os.path.join(self.img_dir, str(img_name))
            image = read_image(img_path)
            images.append(image)
        return images

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        --------
        int
            Number of samples in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        -----------
        idx : int or slice
            Index of the sample to retrieve.

        Returns:
        --------
        tuple
            Tuple containing the image and the corresponding label.
        """
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.img_labels)))
            images, labels = [], []
            for i in indices:
                image, label = self.data[i], self.targets[i]
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
                images.append(image)
                labels.append(label)
            return images, labels
        else:
            image, label = self.data[idx], self.targets[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label


def generate_eurosat(
    classA,
    classB,
    root,
    n_samples=None,
    height=None,
    to_grayscale=False,
    split=True,
    replace=False,
):
    """
    Generate EuroSAT dataset for binary classification between two specified classes.

    Parameters:
    -----------
    classA : int
        Label for the first class.
    classB : int
        Label for the second class.
    root : str
        Root directory of the dataset.
    n_samples : int, optional
        Number of samples to return. If None, all samples are returned (default is None).
    height : int, optional
        Height to resize the images to (preserving aspect ratio by making them square). If None, no resizing is performed (default is None).
    to_grayscale : bool, optional
        If True, convert images to grayscale (default is False).
    split : bool, optional
        If True, return train and test sets separately. If False, return concatenated X and y (default is True).

    Returns:
    --------
    tuple or ndarray
        Depending on the `split` parameter:
        - If split=True (default): Returns tuple (X_train_flat, X_test_flat, y_train, y_test).
        - If split=False: Returns ndarray (X_flat, y).
    """
    # Assuming EuroSAT class is already defined and works similar to torchvision datasets
    dataset_train = EuroSAT(root=root, train=True)
    X_train = torch.stack(dataset_train.data)
    y_train = dataset_train.targets

    dataset_test = EuroSAT(root=root, train=False)
    X_test = torch.stack(dataset_test.data)
    y_test = dataset_test.targets

    # Filter to only include classA and classB
    idxs_train = np.concatenate(
        (np.where(y_train == classA)[0], np.where(y_train == classB)[0])
    )
    idxs_test = np.concatenate(
        (np.where(y_test == classA)[0], np.where(y_test == classB)[0])
    )

    X_train = X_train[idxs_train]
    y_train = y_train[idxs_train]
    X_test = X_test[idxs_test]
    y_test = y_test[idxs_test]

    # Assign -1 and 1 labels
    y_train = np.where(y_train == classA, -1, 1)
    y_test = np.where(y_test == classA, -1, 1)

    if height is not None:
        transform_resize = transforms.Resize((height, height))
        X_train = torch.stack([transform_resize(img) for img in X_train])
        X_test = torch.stack([transform_resize(img) for img in X_test])

    if to_grayscale:
        transform_gray = transforms.Grayscale(num_output_channels=1)
        X_train = torch.stack([transform_gray(img) for img in X_train])
        X_test = torch.stack([transform_gray(img) for img in X_test])

    # Flatten the images
    X_train_flat = X_train.view(X_train.size(0), -1).numpy()
    X_test_flat = X_test.view(X_test.size(0), -1).numpy()

    # Scale the data
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    if n_samples is not None:
        # Randomly select n_samples if specified
        if n_samples > X_train_flat.shape[0]:
            n_samples = X_train_flat.shape[
                0
            ]  # Adjust if n_samples is greater than available samples
        train_indices = np.random.choice(
            X_train_flat.shape[0], n_samples, replace=replace
        )
        test_indices = np.random.choice(
            X_test_flat.shape[0], n_samples, replace=replace
        )
        X_train_flat = X_train_flat[train_indices]
        y_train = y_train[train_indices]
        X_test_flat = X_test_flat[test_indices]
        y_test = y_test[test_indices]

    if split:
        return X_train_flat, X_test_flat, y_train, y_test
    else:
        X_flat = np.concatenate((X_train_flat, X_test_flat), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        return X_flat, y
