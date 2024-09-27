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

# test_model_evaluator.py

import pytest
import logging
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from qcml.bench.model_evaluator import ModelEvaluator

def dummy_transformation(X_train, X_val, y_train, y_val, factor=1):
    X_train_transformed = X_train * factor
    X_val_transformed = X_val * factor
    return X_train_transformed, X_val_transformed, y_train, y_val

def test_evaluate_model():
    X, y = make_classification(n_samples=150, n_features=20, random_state=42)
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]
    evaluator = ModelEvaluator()
    classifier = SVC
    params = {'C': 1.0, 'kernel': 'linear'}
    results = evaluator.evaluate(
        classifier=classifier,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    assert 'accuracy' in results
    assert 'f1_score' in results
    assert 'precision' in results
    assert 'execution_time' in results

def test_evaluate_with_transformation():
    X, y = make_classification(n_samples=150, n_features=20, random_state=42)
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]
    # Apply the transformation before evaluation
    trans_params = {'factor': 0.9}
    X_train_transformed, X_val_transformed, y_train, y_val = dummy_transformation(
        X_train, X_val, y_train, y_val, **trans_params
    )
    evaluator = ModelEvaluator()
    classifier = SVC
    params = {'C': 1.0, 'kernel': 'linear'}
    results = evaluator.evaluate(
        classifier=classifier,
        params=params,
        X_train=X_train_transformed,
        y_train=y_train,
        X_val=X_val_transformed,
        y_val=y_val
    )
    assert 'accuracy' in results

def test_evaluate_invalid_classifier(caplog):
    X, y = make_classification(n_samples=150, n_features=20, random_state=42)
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]
    evaluator = ModelEvaluator()
    classifier = None  # Invalid classifier
    params = {}
    with caplog.at_level(logging.ERROR):
        try:
            evaluator.evaluate(
                classifier=classifier,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
    # Assert that an error was logged
    assert any('Error during model evaluation' in message for message in caplog.messages)

def test_run_with_datasets(monkeypatch):
    # Mock the get_dataset function
    def mock_get_dataset(dataset_name, parameters, **kwargs):
        X, y = make_classification(n_samples=150, n_features=20, random_state=42)
        return X, y

    # Monkeypatch the correct module path
    monkeypatch.setattr('qcml.data.get_dataset', mock_get_dataset)

    # Get the data
    dataset = {'name': 'synthetic_dataset', 'parameters': {}}
    X, y = mock_get_dataset(dataset['name'], dataset['parameters'])
    # Split the data
    from qcml.utils.dataset import split_data
    X_train, X_val, y_train, y_val = split_data(X, y, split_ratio="80/20")

    evaluator = ModelEvaluator()
    classifier = SVC
    params = {'C': 1.0, 'kernel': 'linear'}
    results = evaluator.evaluate(
        classifier=classifier,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    assert 'accuracy' in results

def test_evaluate_with_exception_handling():
    X, y = make_classification(n_samples=150, n_features=20, random_state=42)
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]
    evaluator = ModelEvaluator()
    # Introduce an error by providing incorrect parameters
    classifier = SVC
    params = {'C': -1}  # Invalid value for C
    with pytest.raises(Exception):
        evaluator.evaluate(
            classifier=classifier,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
