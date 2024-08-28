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
import time
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from qcml.bench.helper import (
    log_start_info,
    evaluate_model,
    evaluate_transformed_model,
)
from qcml.bench.kernel_grid import kernel_param_map
from qcml.bench.model_grid import model_grid
from qcml.bench.filter_grid import (
    prepare_param_grid,
    generate_transformation_combinations,
)
from qcml.data import get_dataset
from qcml.utils.dataset import validate_input_data
from qcml.utils.storage import save_results_to_csv
from qcml.bench.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_highest_batch_checkpoint,
    delete_checkpoints,
)

logger = logging.getLogger(__name__)


class GridSearch:
    def __init__(
        self,
        classifier,
        param_grid,
        results_path="results/grid_search/",
        checkpoint_interval=10,
        batch_size=32,
        n_jobs=-1,
        error_traceback=False,
        error_stop=True,
        use_jax=True,
        info_eval_criteria="batch",
        transformation_func=None,
        transformation_params=None,
        experiment_name="default_dataset",
        split_ratio=None,
    ):
        self.classifier = classifier
        self.param_grid = param_grid
        self.results_path = results_path
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.error_traceback = error_traceback
        self.error_stop = error_stop
        self.use_jax = use_jax
        self.info_eval_criteria = info_eval_criteria
        self.transformation_func = transformation_func
        self.transformation_params = transformation_params
        self.experiment_name = experiment_name
        self.split_ratio = split_ratio
        self.results = []
        self.start_batch_idx = 0

    def run(
        self,
        datasets=None,
        X=None,
        y=None,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        return_best=False,
    ):
        try:
            if datasets:
                all_results = []
                for cls, dataset in itertools.product(self.classifier, datasets):
                    self.dataset_name = dataset["name"]
                    dataset_parameters = dataset["parameters"]
                    self.experiment_name_suffix = (
                        "_" + "_".join(dataset_parameters) if dataset_parameters else ""
                    )
                    self.checkpoint_experiment_name = (
                        f"{self.dataset_name}{self.experiment_name_suffix}"
                    )
                    logger.info(
                        f"Starting execution for {cls.__name__} on dataset {self.checkpoint_experiment_name}."
                    )

                    result_filename = f"{cls.__name__}-{self.checkpoint_experiment_name}_best-hypa.csv"
                    result_filepath = os.path.join(self.results_path, result_filename)

                    if os.path.exists(result_filepath):
                        logger.info(
                            f"Skipping execution for {cls.__name__} on {self.checkpoint_experiment_name}. Results already exist."
                        )
                        continue

                    self.start_batch_idx = 0
                    checkpoint_data, batch_idx = get_highest_batch_checkpoint(
                        [cls], self.checkpoint_experiment_name, self.dataset_name
                    )
                    if checkpoint_data:
                        logger.info(
                            f"Resuming from checkpoint {batch_idx} for {cls.__name__} on {self.dataset_name}."
                        )
                        self.results = checkpoint_data["results"]
                        self.start_batch_idx = batch_idx

                    X, y = get_dataset(
                        dataset_name=self.dataset_name, parameters=dataset_parameters
                    )

                    if self.experiment_name == "default_dataset":
                        self.experiment_name = self.checkpoint_experiment_name

                    X_train, X_val, y_train, y_val = validate_input_data(
                        X=X, y=y, split_ratio=self.split_ratio
                    )
                    grid_to_pass = self._prepare_param_grid(cls)
                    result = self._grid_search_single_classifier(
                        cls, grid_to_pass, X_train, y_train, X_val, y_val, return_best
                    )
                    all_results.append(result)
                    delete_checkpoints(
                        [cls], self.checkpoint_experiment_name, self.dataset_name
                    )

                save_results_to_csv(
                    self.results,
                    self.classifier,
                    self.experiment_name,
                    self.results_path,
                )

                return all_results if not return_best else self._get_best_result()

            else:
                self.checkpoint_experiment_name = self.experiment_name
                self.experiment_name_suffix = self.checkpoint_experiment_name
                checkpoint_data, batch_idx = get_highest_batch_checkpoint(
                    self.classifier, self.checkpoint_experiment_name, "custom"
                )
                if checkpoint_data:
                    logger.info(
                        f"Resuming from checkpoint {batch_idx} for {self.classifier.__name__} on custom dataset."
                    )
                    self.results = checkpoint_data["results"]
                    self.start_batch_idx = batch_idx
                else:
                    self.start_batch_idx = 0

                X_train, X_val, y_train, y_val = validate_input_data(
                    X=X,
                    y=y,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    split_ratio=self.split_ratio,
                )
                delete_checkpoints(
                    self.classifier, self.checkpoint_experiment_name, "custom"
                )
                return self._grid_search_single_classifier(
                    self.classifier,
                    self.param_grid,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    return_best,
                )

        finally:
            # TODO - Ensure GPU memory is cleared
            pass

    def _grid_search_single_classifier(
        self, cls, param_grid, X_train, y_train, X_val, y_val, return_best=False
    ):
        transformation_combinations = (
            generate_transformation_combinations(
                self.transformation_func, self.transformation_params
            )
            if self.transformation_func is not None
            else [(None, {})]
        )

        combinations, classifier_name = prepare_param_grid(
            param_grid, kernel_param_map, model_grid, cls, self.info_eval_criteria
        )

        self.n_jobs = (
            min(os.cpu_count(), self.batch_size) if self.n_jobs == -1 else self.n_jobs
        )
        log_start_info(classifier_name, combinations, self.n_jobs)

        best_score, best_params, best_model = self.evaluate_combinations(
            combinations,
            transformation_combinations,
            cls,
            classifier_name,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        save_results_to_csv(
            self.results, classifier_name, self.experiment_name, self.results_path
        )
        delete_checkpoints([cls], self.checkpoint_experiment_name, self.dataset_name)

        if return_best:
            return best_model, best_params, best_score

        return self.results

    def evaluate_combinations(
        self,
        combinations,
        transformation_combinations,
        classifier,
        classifier_name,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        best_score, best_params, best_model = -np.inf, None, None

        total_combinations = len(combinations) * len(transformation_combinations)
        num_batches = int(np.ceil(total_combinations / self.batch_size))
        logger.info(
            f"Total combinations: {total_combinations}, Batch size: {self.batch_size}, Number of batches: {num_batches}"
        )
        logger.debug(f"Logs will be printed by: {self.info_eval_criteria}")

        for batch_idx in range(self.start_batch_idx, num_batches):
            if self.info_eval_criteria == "batch":
                logger.info(f"Starting batch {batch_idx + 1} of {num_batches}")

            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_combinations)
            batch_combinations = combinations[start_idx:end_idx]

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []

                for trans_func, trans_params in transformation_combinations:
                    if trans_func is not None:
                        (
                            X_train_trans,
                            X_val_trans,
                            y_train_trans,
                            y_val_trans,
                        ) = trans_func(X_train, X_val, y_train, y_val, **trans_params)
                    else:
                        X_train_trans, X_val_trans, y_train_trans, y_val_trans = (
                            X_train,
                            X_val,
                            y_train,
                            y_val,
                        )
                    logger.debug(
                        f"Original X shape: {X_train.shape}, Transformed X shape: {X_train_trans.shape}"
                    )
                    for params in batch_combinations:
                        clean_params = {
                            k: v
                            for k, v in params.items()
                            if k not in ["kernel_func", "kernel_params"]
                        }
                        logger.debug(
                            f"Classifier parameters sent to the future: {clean_params}"
                        )
                        futures.append(
                            executor.submit(
                                evaluate_model,
                                clean_params,
                                classifier,
                                X_train_trans,
                                y_train_trans,
                                X_val_trans,
                                y_val_trans,
                                self.use_jax,
                                params.get("kernel_func"),
                                params.get("kernel_params", {}),
                            )
                        )

                for i, future in enumerate(as_completed(futures)):
                    params = None
                    try:
                        (
                            accuracy,
                            f1,
                            precision,
                            execution_time,
                            params,
                            model,
                        ) = future.result()
                        kernel_func = params.pop("kernel_func", None)
                        kernel_params = params.pop("kernel_params", None)
                        transf_func = params.pop("transf_func", None)
                        transf_params = params.pop("transf_params", None)

                        self.results.append(
                            {
                                "experiment_name": self.experiment_name,
                                "classifier": classifier_name,
                                "params": params,
                                "kernel_func": kernel_func,
                                "kernel_params": kernel_params,
                                "accuracy": accuracy,
                                "f1_score": f1,
                                "precision": precision,
                                "execution_time": execution_time,
                                "transf_func": transf_func,
                                "transf_params": transf_params,
                            }
                        )

                        if accuracy > best_score:
                            best_score, best_params, best_model = (
                                accuracy,
                                params,
                                model,
                            )

                        if self.info_eval_criteria == "single":
                            logger.info(
                                f"Evaluation {start_idx + i + 1}/{total_combinations} finished: {params}, time: {execution_time:.2f}s"
                            )
                    except Exception as e:
                        logger.error(f"Error with parameters {params}: {e}")
                        if self.error_traceback:
                            logger.error(traceback.format_exc())
                        if self.error_stop:
                            raise e

            if self.info_eval_criteria == "batch":
                logger.info(
                    f"Batch {batch_idx + 1} of {num_batches} completed ({min(end_idx, total_combinations)}/{total_combinations})."
                )

            if (batch_idx + 1) % self.checkpoint_interval == 0:
                save_checkpoint(
                    results=self.results,
                    classifier_name=[classifier],
                    experiment_name=self.checkpoint_experiment_name,
                    batch_idx=batch_idx + 1,
                    dataset_name=self.dataset_name,
                )

        return best_score, best_params, best_model

    def _prepare_param_grid(self, cls):
        if isinstance(self.param_grid, dict):
            return self.param_grid
        elif (
            isinstance(self.param_grid, list)
            and len(self.param_grid) == len(self.classifier) + 1
        ):
            return [cls, self.param_grid[-1]]
        elif isinstance(self.param_grid, list) and len(self.param_grid) == len(
            self.param_grid
        ):
            return [cls]
        else:
            raise ValueError("Invalid param_grid format")

    def _get_best_result(self):
        if not self.results:
            return None, None, None
        best_result = max(self.results, key=lambda x: x["accuracy"])
        return best_result["classifier"], best_result["params"], best_result["accuracy"]
