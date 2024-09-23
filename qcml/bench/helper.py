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

import itertools
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import time
import logging
import traceback
import multiprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from qcml.bench.kernel_grid import kernel_param_map
from qcml.bench.model_grid import model_grid
from qcml.bench.filter_grid import (
    prepare_param_grid,
    generate_transformation_combinations,
)
from qcml.data import get_dataset
from qcml.utils.kernel import (
    gram_matrix,
    compute_gram_matrix,
    compute_kernel_parallel,
    jitted_gram_matrix,
    jitted_gram_matrix_batched,
)
from qcml.utils.dataset import validate_input_data
from qcml.utils.gpu.info import get_gpu_info
from qcml.utils.gpu.gputil import *
from qcml.utils.storage import save_results_to_csv
from qcml.utils.log import setup_evaluate_model_logging

# multiprocessing.set_start_method("spawn", force=True)
logger = logging.getLogger(__name__)


def log_start_info(classifier_name, combinations, n_jobs):
    logger.info(
        f"Starting grid search with {classifier_name}, {len(combinations)} combinations, {n_jobs} parallel jobs"
    )
    num_gpus, gpu_names = get_gpu_info()
    logger.info(f"GPUs: {num_gpus} available, details: {gpu_names}")


def evaluate_model(
    params,
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    use_jax,
    kernel_func=None,
    kernel_params={},
):
    # logger = setup_evaluate_model_logging(__name__)
    start_time = time.time()

    if use_jax:
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)

    if kernel_func and params.get("kernel") == "precomputed":
        old_X_Train = X_train
        X_train = gram_matrix(
            X1=X_train, X2=X_train, kernel_func=kernel_func, **kernel_params
        )
        X_val = gram_matrix(
            X1=X_val, X2=old_X_Train, kernel_func=kernel_func, **kernel_params
        )

    logger.debug(f"Creating model for: {classifier.__name__}")
    model = classifier(**params)
    logger.debug(f"Fitting: {classifier.__name__}")
    model.fit(np.array(X_train), np.array(y_train))
    logger.debug(f"Predicting: {classifier.__name__}")
    predictions = model.predict(np.array(X_val))

    logger.debug(f"Calculating accuracy score: {classifier.__name__}")
    accuracy = accuracy_score(np.array(y_val), predictions)
    logger.debug(f"Calculating f1 score: {classifier.__name__}")
    f1 = f1_score(np.array(y_val), predictions, average="weighted", zero_division=0)
    logger.debug(f"Calculating precision score: {classifier.__name__}")
    precision = precision_score(
        np.array(y_val), predictions, average="weighted", zero_division=0
    )

    execution_time = time.time() - start_time

    logger.debug(
        f"Model {classifier.__name__} evaluated with params: {params}, accuracy: {accuracy}, f1_score: {f1}, precision: {precision}, execution_time: {execution_time}"
    )

    if kernel_func:
        params["kernel_params"] = kernel_params
        params["kernel_func"] = kernel_func.__name__

    return accuracy, f1, precision, execution_time, params, model


def evaluate_transformed_model(
    params,
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    use_jax,
    trans_func,
    trans_params,
    experiment_name,
):
    # Apply the transformation if a transformation function is provided
    transformation_time = None

    if trans_func is not None:
        start_time = time.time()

        X_train_trans, X_val_trans, y_train_trans, y_val_trans = trans_func(
            X_train, X_val, y_train, y_val, **trans_params
        )

        transformation_time = time.time() - start_time
        logger.debug(
            f"Transformation function '{trans_func.__name__}' took {transformation_time:.4f} seconds with parameters: {trans_params}."
        )
    else:
        X_train_trans, X_val_trans, y_train_trans, y_val_trans = (
            X_train,
            X_val,
            y_train,
            y_val,
        )

    accuracy, f1, precision, execution_time, eval_params, model = evaluate_model(
        {k: v for k, v in params.items() if k not in ["kernel_func", "kernel_params"]},
        classifier,
        X_train_trans,
        y_train_trans,
        X_val_trans,
        y_val_trans,
        use_jax,
        params.get("kernel_func"),
        params.get("kernel_params", {}),
    )

    # Add transformation function name and its parameters to the results
    eval_params["transf_func"] = trans_func.__name__ if trans_func is not None else None
    eval_params["transf_params"] = trans_params if trans_func is not None else {}
    execution_time = (
        (execution_time + transformation_time)
        if transformation_time is not None
        else execution_time
    )

    return accuracy, f1, precision, execution_time, eval_params, model
