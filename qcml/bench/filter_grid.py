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
    """
    Generate all possible combinations of transformation functions and their corresponding parameter grids.

    This function takes a list of transformation functions and a list of corresponding parameter grids,
    and generates all possible combinations of these functions with their parameters. Each combination
    is represented as a tuple containing the function and a dictionary of the parameter values.

    Args:
        transformation_func (list): A list of transformation functions.
        transformation_param_grids (list): A list of dictionaries, where each dictionary represents
                                           the parameter grid for the corresponding transformation function.

    Returns:
        list: A list of tuples. Each tuple contains a transformation function and a dictionary of
              parameter values that correspond to a specific combination of parameters from the grid.
              If no transformation functions are provided, a list containing a single tuple `(None, {})`
              is returned.

    Example:
        >>> def scale(x, factor=1):
        >>>     return x * factor
        >>> def translate(x, offset=0):
        >>>     return x + offset
        >>> transformation_func = [scale, translate]
        >>> transformation_param_grids = [{'factor': [1, 2, 3]}, {'offset': [0, 1]}]
        >>> generate_transformation_combinations(transformation_func, transformation_param_grids)
        >>> # Output: [(<function scale at 0x...>, {'factor': 1}),
                       (<function scale at 0x...>, {'factor': 2}),
                       (<function scale at 0x...>, {'factor': 3}),
                       (<function translate at 0x...>, {'offset': 0}),
                       (<function translate at 0x...>, {'offset': 1})]
    """
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
        # logger.debug(f"- {transformation_combinations}\n")
        return transformation_combinations
    else:
        return [(None, {})]



def prepare_param_grid(
    param_grid, kernel_param_map, model_grid, classifier, log_combinations
):
    """
    Prepare a parameter grid for a given classifier and its associated kernel parameters.

    This function takes a parameter grid, a map of kernel parameters, a model grid, and a classifier,
    and prepares all valid combinations of these parameters. It supports different formats of `param_grid`,
    handling both dictionary and list inputs. The function returns the prepared combinations and the name
    of the classifier.

    Args:
        param_grid (dict or list): The parameter grid, either as a dictionary or a list.
                                   If a list, it should contain the classifier and optionally the kernel parameter grid.
        kernel_param_map (dict): A dictionary mapping kernel functions to their respective parameter grids.
        model_grid (dict): A dictionary mapping classifier names to their respective parameter grids.
        classifier (object): The classifier for which the parameter grid is being prepared.
        log_combinations (bool): If True, the valid combinations are logged.

    Returns:
        tuple: A tuple containing:
            - combinations (list): A list of valid parameter combinations.
            - classifier_name (str): The name of the classifier.

    Raises:
        ValueError: If `param_grid` is neither a dictionary nor a list with the expected format.

    Example:
        >>> classifier = SVC
        >>> param_grid = [{'kernel_func': some_kernel_func}, {'kernel_params': [1, 2]}]
        >>> kernel_param_map = {'some_kernel_func': [1, 2]}
        >>> model_grid = {'SVC': {'C': [1, 10], 'gamma': [0.1, 0.01]}}
        >>> prepare_param_grid(param_grid, kernel_param_map, model_grid, classifier, log_combinations=True)
        >>> # Output: (list_of_combinations, 'SVC')
    """
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

    # Log the debug information
    logger.debug(
        f"Prepared {len(combinations)} combinations for classifier '{classifier_name}'."
    )

    return combinations, classifier_name



def filter_valid_combinations(
    param_grid, kernel_param_map=None, log_combinations=False
):
    """
    Filter and validate parameter combinations from a parameter grid.

    This function generates all possible combinations of parameters from a provided parameter grid.
    It validates these combinations based on the presence of required kernel parameters, filters out
    duplicates, and optionally logs valid and invalid combinations. The function returns a list of valid
    combinations.

    Args:
        param_grid (dict): A dictionary where keys are parameter names and values are lists of possible values.
        kernel_param_map (dict, optional): A dictionary mapping kernel function names to a list of required parameter keys.
                                           Defaults to None.
        log_combinations (bool, optional): If True, logs the valid and invalid combinations. Defaults to False.

    Returns:
        list: A list of dictionaries, where each dictionary represents a valid combination of parameters.

    Example:
        >>> param_grid = {
        >>>     'C': [1, 10],
        >>>     'kernel_func': [some_kernel_func],
        >>>     'kernel_params': [{'param1': 0.1}, {'param2': 0.2}]
        >>> }
        >>> kernel_param_map = {'some_kernel_func': ['param1', 'param2']}
        >>> filter_valid_combinations(param_grid, kernel_param_map)
        >>> # Output: [{'C': 1, 'kernel_func': some_kernel_func, 'kernel_params': {'param1': 0.1}},
        >>> #          {'C': 10, 'kernel_func': some_kernel_func, 'kernel_params': {'param2': 0.2}}]
    """
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
                if log_combinations:
                    pass#logger.debug(f"Invalid combination due to missing kernel params: {params}")
            elif all(param in kernel_param_keys for param in kernel_params.keys()):
                valid_combinations.append(params)
                if log_combinations:
                    pass#logger.debug(f"Valid combination: {params}")
            else:
                invalid_combinations += 1
                if log_combinations:
                    pass#logger.debug(f"Invalid combination due to incorrect kernel params: {params}")

    logger.info(
        f"Total combinations: {total_combinations}, Valid combinations: {len(valid_combinations)}, Invalid combinations: {invalid_combinations}"
    )
    return valid_combinations
