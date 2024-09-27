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

# test_parameter_grid.py

import pytest
from qcml.bench.parameter_grid import ParameterGrid

def dummy_transformation(X_train, X_val, y_train, y_val, factor=1):
    # Simple dummy transformation
    return X_train * factor, X_val * factor, y_train, y_val

def test_parameter_grid_combinations():
    param_grid = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
    transformations = [dummy_transformation]
    transformation_params = [{'factor': [0.8, 1.0, 1.2]}]
    pg = ParameterGrid(param_grid, transformations, transformation_params)
    combinations = pg.combinations
    # Expected combinations: 2 (C) * 2 (kernel) * 1 (transformation) * 3 (factor) = 12
    expected_combinations = 2 * 2 * 1 * 3
    assert len(combinations) == expected_combinations

def test_parameter_grid_no_transformations():
    param_grid = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
    pg = ParameterGrid(param_grid)
    combinations = pg.combinations
    # Expected combinations: 2 (C) * 2 (kernel) = 4
    assert len(combinations) == 4

def test_parameter_grid_no_params():
    # Test with empty parameter grid
    param_grid = {}
    transformations = [dummy_transformation]
    transformation_params = [{'factor': [1]}]
    pg = ParameterGrid(param_grid, transformations, transformation_params)
    combinations = pg.combinations
    # Expected combinations: 1 (no params) * 1 (transformation) * 1 (factor) = 1
    assert len(combinations) == 1

def test_run_with_datasets(monkeypatch):
    from qcml.data import get_dataset

    # Mock the get_dataset function
    def mock_get_dataset(dataset_name, parameters):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=150, n_features=20, random_state=42)
        return X, y

    # Monkeypatch the correct module path
    monkeypatch.setattr('qcml.data.get_dataset', mock_get_dataset)

    param_grid = {'C': [1]}
    transformations = [dummy_transformation]
    transformation_params = [{'factor': [0.8, 1.0, 1.2]}]
    datasets = [{'name': 'synthetic_dataset', 'parameters': {}}]
    pg = ParameterGrid(param_grid, transformations, transformation_params)
    combinations = pg.combinations
    # Expected combinations: 1 (C) * 1 (transformation) * 3 (factor) = 3
    assert len(combinations) == 3

def test_parameter_grid_invalid_params():
    # Test with invalid parameters
    param_grid = {'C': [-1]}  # Invalid value for C
    pg = ParameterGrid(param_grid)
    combinations = pg.combinations
    # Even though the parameter is invalid, combinations are generated
    assert len(combinations) == 1
