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

# parameter_grid.py

import itertools
from typing import Optional, List, Callable, Dict, Any, Tuple

class ParameterGrid:
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        transformations: Optional[List[Callable]] = None,
        transformation_params: Optional[List[Dict[str, List[Any]]]] = None,
    ):
        self.param_grid = param_grid
        self.transformations = transformations or []
        self.transformation_params = transformation_params or []
        self.combinations = self._create_param_combinations()

    def _create_param_combinations(self) -> List[Tuple[Dict[str, Any], Optional[Callable], Dict[str, Any]]]:
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = [
            dict(zip(param_names, values))
            for values in itertools.product(*param_values)
        ]

        # Generate transformation combinations
        transformation_combinations = self._generate_transformation_combinations()

        # Combine parameter and transformation combinations
        full_combinations = [
            (params, trans_func, trans_params)
            for params in param_combinations
            for trans_func, trans_params in transformation_combinations
        ]

        return full_combinations

    def _generate_transformation_combinations(self) -> List[Tuple[Optional[Callable], Dict[str, Any]]]:
        if not self.transformations or not self.transformation_params:
            return [(None, {})]
        transformation_combinations = []
        for func, param_grid in zip(self.transformations, self.transformation_params):
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            for combination in itertools.product(*param_values):
                transformation_combinations.append(
                    (func, dict(zip(param_names, combination)))
                )
        return transformation_combinations
