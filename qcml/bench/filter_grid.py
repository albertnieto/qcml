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
import logging
from qcml.utils.types import convert_to_hashable

logger = logging.getLogger(__name__)

from itertools import product


def generate_transformation_combinations(
    transformation_func, transformation_param_grids
):
    if transformation_func is not None:
        transformation_combinations = []

        for func, param_grid in zip(transformation_func, transformation_param_grids):
            param_names = param_grid.keys()
            param_values = param_grid.values()
            for combination in product(*param_values):
                transformation_combinations.append(
                    (func, dict(zip(param_names, combination)))
                )

        logger.info(
            f"Generated {len(transformation_combinations)} transformation combinations."
        )
        logger.debug(f"- {transformation_combinations}\n")
        return transformation_combinations
    else:
        return [(None, {})]


def prepare_param_grid(
    param_grid, kernel_param_map, model_grid, classifier, log_combinations
):
    if isinstance(param_grid, dict):
        combinations = filter_valid_combinations(
            param_grid, kernel_param_map, log_combinations
        )
    elif isinstance(param_grid, list) and len(param_grid) == 2:
        classifier = param_grid[0]
        kernel_param_grid = param_grid[1]
        classifier_name = getattr(classifier, "__name__", str(classifier))
        param_grid = model_grid.get(classifier_name, {})
        param_grid["kernel_func"] = [kp["kernel_func"] for kp in kernel_param_grid]
        param_grid["kernel_params"] = [kp["kernel_params"] for kp in kernel_param_grid]
        combinations = filter_valid_combinations(
            param_grid, kernel_param_map, log_combinations
        )
    elif isinstance(param_grid, list) and len(param_grid) == 1:
        classifier = param_grid[0]
        classifier_name = getattr(classifier, "__name__", str(classifier))
        param_grid = model_grid.get(classifier_name, {})
        combinations = filter_valid_combinations(param_grid)
    else:
        raise ValueError(
            "param_grid should either be a dictionary or a list with two elements: [classifier, kernel_param_grid]."
        )

    return combinations, classifier_name


def filter_valid_combinations(
    param_grid, kernel_param_map=None, log_combinations=False
):
    valid_combinations = []
    invalid_combinations = 0
    seen_combinations = set()

    keys, values = zip(*param_grid.items())
    total_combinations = len(list(itertools.product(*values)))

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        params_tuple = tuple(
            sorted((k, convert_to_hashable(v)) for k, v in params.items())
        )

        if params_tuple in seen_combinations:
            continue

        seen_combinations.add(params_tuple)
        kernel_func = params.get("kernel_func")

        if kernel_func is None:
            valid_combinations.append(params)
            logger.debug(f"Valid combination: {params}")
        else:
            kernel_func_name = kernel_func.__name__
            kernel_params = params.get("kernel_params", {})

            kernel_param_keys = kernel_param_map.get(kernel_func_name, [])
            if kernel_param_keys and not kernel_params:
                invalid_combinations += 1
                # if log_combinations:
                # logger.debug(f"Invalid combination due to missing kernel params: {params}")
            elif all(param in kernel_param_keys for param in kernel_params.keys()):
                valid_combinations.append(params)
                # logger.debug(f"Valid combination: {params}")
            else:
                invalid_combinations += 1
                # logger.debug(f"Invalid combination due to incorrect kernel params: {params}")

    logger.info(
        f"Total combinations: {total_combinations}, Valid combinations: {len(valid_combinations)}, Invalid combinations: {invalid_combinations}"
    )
    # valid_combinations_str = "\n".join([str(combination) for combination in valid_combinations])
    # logger.debug(f"Valid combinations: {valid_combinations_str}")
    return valid_combinations
