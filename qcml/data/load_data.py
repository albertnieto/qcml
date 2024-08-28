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
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def get_dataset(
    dataset_name,
    root_path="datasets/",
    file_type="join",
    parameters=None,
    split_ratio=None,
):
    """
    Read data from CSV files where each row is a data sample.
    The columns are the input features, and the last column specifies a label.

    If file_type is 'join', the function checks for both 'train' and 'test' files. If both are found, it concatenates the data.
    If file_type is 'split', the function will return X_train, X_test, y_train, y_test.
    If just one file exists, the data will be split according to split_ratio.

    Args:
        dataset_name (str): Name of the dataset. This is a required parameter.
        root_path (str, optional): Root directory of the datasets. Defaults to 'datasets/'.
        file_type (str, optional): Type of the file handling, either 'join', 'split', 'train', or 'test'. Defaults to 'join'.
        parameters (list, optional): Additional parameters to append to the file name, delimited by '_'. Defaults to None.
        split_ratio (str, optional): Split ratio for train/test split, e.g., '80/20'. Only used if file_type is 'split'.

    Returns:
        X (ndarray): 2-d array of inputs.
        y (ndarray): Array of labels.
        If file_type is 'split', returns X_train, X_test, y_train, y_test.
    """
    if parameters:
        params = "_".join(parameters)
    else:
        params = ""

    def load_file(file_type=None):
        # Build the filename based on the dataset name, file_type, and additional parameters
        file_name = f"{dataset_name}"
        if params:
            file_name = f"{file_name}_{params}"
        if file_type:
            file_name = f"{file_name}_{file_type}"
        file_name = f"{file_name}.csv"

        # Construct the full file path
        file_path = os.path.join(root_path, dataset_name, file_name)

        try:
            logging.debug(f"Attempting to load file: {file_path}")
            # Attempt to read the CSV file
            data = pd.read_csv(file_path, header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            return X, y
        except Exception as e:
            # Log any other exceptions that may occur
            logging.error(f"An error occurred while reading the file: {file_path}")
            logging.error(e)
            return None, None

    if file_type == "join":
        logging.debug("File type is 'join'. Checking for both train and test files.")
        # Check if both train and test files exist
        X_train, y_train = load_file("train")
        X_test, y_test = load_file("test")

        if X_train is not None and X_test is not None:
            logging.debug("Both train and test files found. Concatenating data.")
            # If both train and test files exist, concatenate them
            X = np.vstack((X_train, X_test))
            y = np.hstack((y_train, y_test))
            return X, y
        elif X_train is not None:
            logging.debug("Only train file found. Returning train data.")
            return X_train, y_train
        elif X_test is not None:
            logging.debug("Only test file found. Returning test data.")
            return X_test, y_test
        else:
            logging.debug(
                "Neither train nor test files found. Trying without file_type."
            )
            # If neither train nor test files are found, try loading without file_type
            return load_file(None)

    elif file_type == "split":
        logging.debug("File type is 'split'. Checking for train and test files.")
        # Check if both train and test files exist
        X_train, y_train = load_file("train")
        X_test, y_test = load_file("test")

        if X_train is not None and X_test is not None:
            logging.debug("Both train and test files found. Returning separate data.")
            return X_train, X_test, y_train, y_test
        else:
            logging.debug(
                "Either train or test file is missing. Attempting to split from a single file."
            )
            # If only one file exists, perform the split based on split_ratio
            X, y = load_file(None)
            if X is not None and split_ratio is not None:
                train_ratio, test_ratio = map(float, split_ratio.split("/"))
                train_size = int(len(X) * (train_ratio / (train_ratio + test_ratio)))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                logging.debug(f"Data split with ratio {split_ratio}.")
                return X_train, X_test, y_train, y_test
            else:
                logging.error(
                    "Failed to load data for splitting or invalid split ratio provided."
                )
                return None, None, None, None

    else:
        logging.debug(f"File type is '{file_type}'. Loading specified type.")
        # Load only the specified file type ('train' or 'test')
        return load_file(file_type)
