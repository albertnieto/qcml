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

# test_grid_search.py

import pytest
import numpy as np
import logging
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from qcml.bench.grid_search import GridSearch

def dummy_transformation(X_train, X_val, y_train, y_val, factor=1):
    X_train_transformed = X_train * factor
    X_val_transformed = X_val * factor
    return X_train_transformed, X_val_transformed, y_train, y_val

@pytest.fixture
def data():
    X, y = make_classification(n_samples=150, n_features=20, random_state=42)
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]
    return X_train, y_train, X_val, y_val

def test_run_without_datasets(data):
    X_train, y_train, X_val, y_val = data
    classifiers = [SVC]
    param_grid = {'C': [1.0], 'kernel': ['linear']}
    grid_search = GridSearch(
        classifiers=classifiers,
        param_grid=param_grid,
        experiment_name='test_experiment'
    )
    results = grid_search.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    assert isinstance(results, list)
    assert len(results) == 1  # One classifier
    assert len(grid_search.results) == 1  # One combination
    assert 'accuracy' in grid_search.results[0]

def test_run_with_transformations(data):
    X_train, y_train, X_val, y_val = data
    classifiers = [SVC]
    param_grid = {'C': [1.0], 'kernel': ['linear']}
    transformations = [dummy_transformation]
    transformation_params = [{'factor': [0.8, 1.0]}]
    grid_search = GridSearch(
        classifiers=classifiers,
        param_grid=param_grid,
        transformations=transformations,
        transformation_params=transformation_params,
        experiment_name='test_experiment'
    )
    results = grid_search.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    assert isinstance(results, list)
    assert len(results) == 1  # One classifier
    expected_combinations = 2  # 1 param combination * 2 transformations
    assert len(grid_search.results) == expected_combinations

def test_run_with_return_best(data):
    X_train, y_train, X_val, y_val = data
    classifiers = [SVC]
    param_grid = {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearch(
        classifiers=classifiers,
        param_grid=param_grid,
        experiment_name='test_experiment'
    )
    best_model, best_params, best_score = grid_search.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        return_best=True
    )
    assert best_model is not None
    assert best_params is not None
    assert isinstance(best_score, float)

def test_run_invalid_classifier(data, caplog):
    X_train, y_train, X_val, y_val = data
    classifiers = [None]  # Invalid classifier
    param_grid = {'C': [1.0]}
    grid_search = GridSearch(
        classifiers=classifiers,
        param_grid=param_grid,
        experiment_name='test_experiment',
        error_stop=False  # To prevent the test from stopping
    )
    with caplog.at_level(logging.ERROR):
        grid_search.run(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
    # Assert that an error was logged
    assert any('Error with parameters' in message for message in caplog.messages)
    # Assert that results are empty
    assert len(grid_search.results) == 0

def test_run_with_datasets(monkeypatch):
    classifiers = [SVC]
    param_grid = {'C': [1.0], 'kernel': ['linear']}
    datasets = [{'name': 'synthetic_dataset', 'parameters': {}}]

    # Mock the get_dataset function
    def mock_get_dataset(dataset_name, parameters):
        X, y = make_classification(n_samples=150, n_features=20, random_state=42)
        return X, y

    # Replace get_dataset with the mock function in qcml.bench.grid_search
    monkeypatch.setattr('qcml.bench.grid_search.get_dataset', mock_get_dataset)

    grid_search_instance = GridSearch(
        classifiers=classifiers,
        param_grid=param_grid,
        experiment_name='test_experiment',
        split_ratio="80/20"  # Provide split_ratio to avoid ValueError
    )
    results = grid_search_instance.run(datasets=datasets)
    assert isinstance(results, list)
    assert len(results) == 1  # One classifier
