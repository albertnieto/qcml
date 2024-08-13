import itertools
import logging
from ..utils.types import convert_to_hashable

logger = logging.getLogger(__name__)

def filter_valid_combinations(param_grid, kernel_param_map, log_combinations):
    valid_combinations = []
    invalid_combinations = 0
    seen_combinations = set()
    
    keys, values = zip(*param_grid.items())
    total_combinations = len(list(itertools.product(*values)))
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        params_tuple = tuple(sorted((k, convert_to_hashable(v)) for k, v in params.items()))
        
        if params_tuple in seen_combinations:
            continue
        
        seen_combinations.add(params_tuple)
        kernel_func = params.get('kernel_func')
        
        if kernel_func is None:
            valid_combinations.append(params)
            logger.debug(f"Valid combination: {params}")
        else:
            kernel_func_name = kernel_func.__name__
            kernel_params = params.get('kernel_params', {})
            
            kernel_param_keys = kernel_param_map.get(kernel_func_name, [])
            if kernel_param_keys and not kernel_params:
                invalid_combinations += 1
                if log_combinations:
                    logger.debug(f"Invalid combination due to missing kernel params: {params}")
            elif all(param in kernel_param_keys for param in kernel_params.keys()):
                valid_combinations.append(params)
                logger.debug(f"Valid combination: {params}")
            else:
                invalid_combinations += 1
                logger.debug(f"Invalid combination due to incorrect kernel params: {params}")

    logger.info(f"Total combinations: {total_combinations}, Valid combinations: {len(valid_combinations)}, Invalid combinations: {invalid_combinations}")
    return valid_combinations