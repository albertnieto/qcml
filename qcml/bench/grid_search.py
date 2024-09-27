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
import itertools
import logging
import traceback
from typing import Optional, List, Callable, Dict, Any, Tuple

from qcml.bench.parameter_grid import ParameterGrid
from qcml.bench.model_evaluator import ModelEvaluator
from qcml.utils.storage import save_results_to_csv
from qcml.utils.dataset import validate_input_data
from qcml.data import get_dataset

logger = logging.getLogger(__name__)


class GridSearch:
    def __init__(
        self,
        classifiers: List[Callable],
        param_grid: Dict[str, List[Any]],
        combinations: Optional[
            List[Tuple[Dict[str, Any], Callable, Dict[str, Any]]]
        ] = None,
        results_path: str = "results/grid_search/",
        batch_size: int = 32,
        n_jobs: int = -1,
        error_traceback: bool = False,
        error_stop: bool = True,
        use_jax: bool = False,
        log_level: str = "batch",
        transformations: Optional[List[Callable]] = None,
        transformation_params: Optional[List[Dict[str, List[Any]]]] = None,
        experiment_name: str = "default_dataset",
        split_ratio: Optional[float] = None,
    ):
        self.classifiers = classifiers
        self.param_grid = param_grid
        self.combinations = combinations
        self.results_path = results_path
        self.batch_size = batch_size
        self.n_jobs = min(os.cpu_count(), batch_size) if n_jobs == -1 else n_jobs
        self.error_traceback = error_traceback
        self.error_stop = error_stop
        self.use_jax = use_jax
        self.log_level = log_level
        self.transformations = transformations
        self.transformation_params = transformation_params
        self.experiment_name = experiment_name
        self.split_ratio = split_ratio
        self.results = []

        # Initialize ModelEvaluator
        self.evaluator = ModelEvaluator(use_jax=self.use_jax)

        # Initialize ParameterGrid if combinations are not provided
        if self.combinations is None:
            self.param_grid_obj = ParameterGrid(
                self.param_grid, self.transformations, self.transformation_params
            )
            self.combinations = self.param_grid_obj.combinations

    def run(
        self,
        datasets: Optional[List[Dict[str, Any]]] = None,
        X=None,
        y=None,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        return_best: bool = False,
    ):
        if not self.use_jax:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        all_results = []

        if datasets:
            for cls, dataset in itertools.product(self.classifiers, datasets):
                result = self._process_dataset(
                    cls, dataset, X_train, y_train, X_val, y_val, return_best
                )
                all_results.append(result)
        else:
            for cls in self.classifiers:
                result = self._process_dataset(
                    cls, None, X_train, y_train, X_val, y_val, return_best, X, y
                )
                all_results.append(result)

        return self._select_best_model() if return_best else all_results

    def _process_dataset(
        self,
        cls,
        dataset: Optional[Dict[str, Any]],
        X_train,
        y_train,
        X_val,
        y_val,
        return_best: bool,
        X=None,
        y=None,
    ):
        if dataset:
            dataset_name = dataset["name"]
            dataset_params = dataset.get("parameters", {})
            experiment_suffix = (
                "_" + "_".join(str(v) for v in dataset_params.values())
                if dataset_params
                else ""
            )
            experiment_full_name = f"{dataset_name}{experiment_suffix}"
            logger.info(
                f"Starting execution for {cls.__name__} on dataset {experiment_full_name}."
            )

            result_filename = f"{cls.__name__}-{experiment_full_name}_best-hypa.csv"
            result_filepath = os.path.join(self.results_path, result_filename)

            if os.path.exists(result_filepath):
                logger.info(
                    f"Skipping execution for {cls.__name__} on {experiment_full_name}. Results already exist."
                )
                return

            # Replace with your own data loading function if necessary
            X, y = get_dataset(dataset_name=dataset_name, parameters=dataset_params)

            if self.experiment_name == "default_dataset":
                self.experiment_name = experiment_full_name

        # Validate and split the data
        X_train, X_val, y_train, y_val = validate_input_data(
            X=X,
            y=y,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            split_ratio=self.split_ratio,
        )

        classifier_name = getattr(cls, "__name__", cls.__class__.__name__)
        logger.info(
            f"Starting grid search with {classifier_name}, {len(self.combinations)} combinations, {self.n_jobs} parallel jobs"
        )

        best_score, best_params, best_model = self._evaluate_combinations(
            cls, classifier_name, X_train, y_train, X_val, y_val
        )

        # Save results to CSV
        save_results_to_csv(
            self.results,
            classifier_name,
            self.experiment_name,
            self.results_path,
        )

        return (best_model, best_params, best_score) if return_best else self.results

    def _evaluate_combinations(
        self,
        classifier,
        classifier_name: str,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        best_score = -float("inf")
        best_params = None
        best_model = None
        self.results = []
        total_combinations = len(self.combinations)

        for idx, (params, trans_func, trans_params) in enumerate(self.combinations):
            try:
                evaluation_result = self.evaluator.evaluate(
                    classifier,
                    params,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    trans_func,
                    trans_params,
                )

                self.results.append(
                    {
                        "experiment_name": self.experiment_name,
                        "classifier": classifier_name,
                        "params": params,
                        "accuracy": evaluation_result["accuracy"],
                        "f1_score": evaluation_result["f1_score"],
                        "precision": evaluation_result["precision"],
                        "execution_time": evaluation_result["execution_time"],
                        "transf_func": trans_func.__name__ if trans_func else None,
                        "transf_params": trans_params if trans_func else None,
                    }
                )

                if evaluation_result["accuracy"] > best_score:
                    best_score = evaluation_result["accuracy"]
                    best_params = params
                    best_model = evaluation_result["model"]

                if self.log_level == "single":
                    logger.info(
                        f"Evaluated params: {params}, accuracy: {evaluation_result['accuracy']:.4f}"
                    )

            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                if self.error_traceback:
                    logger.error(traceback.format_exc())
                if self.error_stop:
                    raise e

            if self.log_level == "batch" and (idx + 1) % self.batch_size == 0:
                logger.info(f"Processed {idx + 1}/{total_combinations} combinations.")

        return best_score, best_params, best_model

    def _select_best_model(self):
        if not self.results:
            return None, None, None
        best_result = max(self.results, key=lambda x: x["accuracy"])
        return best_result["classifier"], best_result["params"], best_result["accuracy"]
