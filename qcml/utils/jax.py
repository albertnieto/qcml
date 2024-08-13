import os
import logging
from jax.lib import xla_bridge
import jax

def jax_setup(use_cpu=False, mem_fraction=0.4, gpu_index=0, check_tracer_leaks=False):
    """
    Set up JAX configuration based on provided parameters.

    Parameters:
    - use_cpu (bool): If True, use CPU instead of GPU. Default is False.
    - mem_fraction (float): Fraction of GPU memory to use. Default is 0.4.
    - gpu_index (int): Index of the GPU to use. Default is 0.
    - check_tracer_leaks (bool): If True, enables checking for tracer leaks. Default is False.

    Returns:
    None
    """
    try:       
        logger = logging.getLogger(__name__)
        
        # Use CPU instead if specified
        if use_cpu:
            jax.config.update('jax_platform_name', 'cpu')
            logger.info("Configured to use CPU.")
        else:
            jax.config.update('jax_platform_name', 'gpu')
            logger.info("Configured to use GPU.")

        # Limit memory usage to a fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        logger.info(f"Memory fraction set to {mem_fraction}.")

        # Select specified GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        logger.info(f"CUDA_VISIBLE_DEVICES set to {gpu_index}.")

        # Set JAX_CHECK_TRACER_LEAKS based on parameter
        os.environ['JAX_CHECK_TRACER_LEAKS'] = 'true' if check_tracer_leaks else 'false'
        logger.info(f"JAX_CHECK_TRACER_LEAKS set to {'true' if check_tracer_leaks else 'false'}.")

        # Get and print backend platform
        backend = xla_bridge.get_backend().platform
        logger.info(f"Backend in use: {backend}")

        # Get and print GPU devices
        gpus = jax.devices('gpu')
        if gpus:
            logger.info("GPUs used:")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
        else:
            logger.info("No GPUs available.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    jax_setup(use_cpu=False, mem_fraction=0.5, gpu_index=0, check_tracer_leaks=True)
