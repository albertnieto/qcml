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

# model_evaluator.py

import time
from typing import Optional, Callable, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score

class ModelEvaluator:
    def __init__(self, use_jax: bool = False):
        self.use_jax = use_jax

    def evaluate(
        self,
        classifier,
        params: Dict[str, Any],
        X_train,
        y_train,
        X_val,
        y_val,
        trans_func: Optional[Callable] = None,
        trans_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Apply transformation if provided
        if trans_func:
            X_train, X_val, y_train, y_val = trans_func(
                X_train, X_val, y_train, y_val, **trans_params
            )

        # Train and evaluate the model
        model = classifier(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average="weighted", zero_division=0)
        precision = precision_score(y_val, predictions, average="weighted", zero_division=0)

        execution_time = time.time() - start_time

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "execution_time": execution_time,
            "model": model,
        }
