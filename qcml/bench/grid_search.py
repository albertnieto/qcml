import itertools
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import time
import logging
import traceback
import GPUtil

from sklearn.metrics import accuracy_score, f1_score, precision_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from .kernel_grid import kernel_param_map
from .model_grid import model_grid
from .filter_grid import filter_valid_combinations 
from ..utils.kernel import compute_gram_matrix
from ..utils.dataset import split_data
from ..utils.gpu.info import get_gpu_info

logger = logging.getLogger(__name__)

def evaluate_model(params, classifier, X_train, y_train, X_val, y_val, use_jax, kernel_func=None, kernel_params={}):
    start_time = time.time()
    
    # Convert to JAX arrays if use_jax is True
    if use_jax:
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)

    if kernel_func:
        # Compute Gram matrix for the training set
        X_train = compute_gram_matrix(X_train, X_train, kernel_func, **kernel_params)
        
        # Compute Gram matrix for the validation set with respect to the training set
        X_val = compute_gram_matrix(X_val, X_train, kernel_func, **kernel_params)
        
        # Set kernel as precomputed in params
        params['kernel'] = 'precomputed'
    
    # Initialize and train the model
    model = classifier(**params)
    model.fit(np.array(X_train), np.array(y_train))

    # Make predictions on the validation set
    predictions = model.predict(np.array(X_val))
    
    # Compute accuracy, F1 score, and precision
    accuracy = accuracy_score(np.array(y_val), predictions)
    f1 = f1_score(np.array(y_val), predictions, average='weighted', zero_division=0)
    precision = precision_score(np.array(y_val), predictions, average='weighted', zero_division=0)

    # Calculate execution time
    execution_time = time.time() - start_time

    logger.debug(f"Model evaluated with params: {params}, accuracy: {accuracy}, f1_score: {f1}, precision: {precision}, execution_time: {execution_time}")

    # Store kernel parameters and function name if a custom kernel was used
    if kernel_func:
        params['kernel_params'] = kernel_params
        params['kernel_func'] = kernel_func.__name__

    return accuracy, f1, precision, execution_time, params, model


def grid_search(classifier, param_grid, kernel_param_map=kernel_param_map, model_grid=model_grid, X=None, y=None, X_train=None, y_train=None, X_val=None, y_val=None, use_jax=True, n_jobs=-1, results_path="results/", experiment_name="tfm", split_ratio=None, error_traceback=False, log_combinations=False):
    if (X is not None and y is None) or (X is None and y is not None):
        raise ValueError("Both X and y must be provided if using whole dataset.")

    if (X_train is not None and y_train is None) or (X_train is None and y_train is not None):
        raise ValueError("Both X_train and y_train must be provided if using pre-split dataset.")

    if (X_val is not None and y_val is None) or (X_val is None and y_val is not None):
        raise ValueError("Both X_val and y_val must be provided if using pre-split dataset.")

    if X is not None and split_ratio is None:
        raise ValueError("Split ratio must be provided if using whole dataset.")

    if X is not None and y is not None:
        X_train, X_val, y_train, y_val = split_data(X, y, split_ratio)
        logger.info(f"Data split into training and validation sets using split ratio: {split_ratio}")
    elif X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError("Insufficient data provided. Provide either whole dataset or pre-split dataset.")

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []

    if isinstance(param_grid, dict):
        combinations = filter_valid_combinations(param_grid, kernel_param_map, log_combinations)
    elif isinstance(param_grid, list) and len(param_grid) == 2:
        classifier = param_grid[0]
        kernel_param_grid = param_grid[1]
        classifier_name = getattr(classifier, '__name__', str(classifier))
        param_grid = model_grid.get(classifier_name, {})
        param_grid['kernel_func'] = [kp['kernel_func'] for kp in kernel_param_grid]
        param_grid['kernel_params'] = [kp['kernel_params'] for kp in kernel_param_grid]
        combinations = filter_valid_combinations(param_grid, kernel_param_map, log_combinations)
    else:
        raise ValueError("param_grid should either be a dictionary or a list with two elements: [classifier, kernel_param_grid].")

    n_jobs = n_jobs if n_jobs != -1 else None
    logger.info(f"Starting grid search with classifier: {classifier_name}, n_jobs: {n_jobs}, number of combinations: {len(combinations)}")

    num_gpus, gpu_names = get_gpu_info()
    logger.info(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 0:
        logger.info(f"GPU details: {gpu_names}")

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                evaluate_model, 
                {k: v for k, v in params.items() if k not in ['kernel_func', 'kernel_params']},
                classifier, 
                X_train, 
                y_train, 
                X_val, 
                y_val, 
                use_jax, 
                params.get('kernel_func', None), 
                params.get('kernel_params', {})
            ) 
            for params in combinations
        ]

        num_parallel_evaluations = len(futures)
        logger.info(f"Number of parallel evaluations: {num_parallel_evaluations}")

        for i, future in enumerate(as_completed(futures)):
            params = None
            try:
                accuracy, f1, precision, execution_time, params, model = future.result()
                if 'kernel_params' in params:
                    params.update(params.pop('kernel_params'))
                if 'kernel_func' in params:
                    params['kernel_func'] = params['kernel_func']

                results.append({
                    "experiment_name": experiment_name,
                    "classifier": classifier_name,
                    "params": params,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "execution_time": execution_time
                })
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = params
                    best_model = model
                logger.info(f"Evaluation {i+1}/{num_parallel_evaluations} finished for params: {params}, execution time: {execution_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")
                if error_traceback:
                    logger.error(traceback.format_exc())

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    csv_file_name = f"{classifier_name}-{experiment_name}_best-hypa.csv"
    csv_file_path = os.path.join(results_path, csv_file_name)

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_file_path, index=False)
    logger.debug(f"Results saved to {csv_file_path}, file size: {os.path.getsize(csv_file_path)} bytes")

    return best_model, best_params, best_score
