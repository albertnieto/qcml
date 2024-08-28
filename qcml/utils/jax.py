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
import logging
from jax.lib import xla_bridge
import jax.numpy as jnp
import jax
import logging

logger = logging.getLogger(__name__)


def jax_setup(
    use_cpu=False,
    mem_fraction=0.1,
    gpu_index=0,
    check_tracer_leaks=False,
    xla_performance_flags=True,
):
    """
    Set up JAX configuration based on provided parameters.

    Parameters:
    - use_cpu (bool): If True, use CPU instead of GPU. Default is False.
    - mem_fraction (float): Fraction of GPU memory to use. Default is 0.4.
    - gpu_index (int): Index of the GPU to use. Default is 0.
    - check_tracer_leaks (bool): If True, enables checking for tracer leaks. Default is False.
    - xla_performance_flags (bool): If True, enables XLA performance optimizations. Default is True.

    Returns:
    None
    """
    try:
        logger = logging.getLogger(__name__)

        # Use CPU instead if specified
        if use_cpu:
            jax.config.update("jax_platform_name", "cpu")
            logger.info("Configured to use CPU.")
        else:
            jax.config.update("jax_platform_name", "gpu")
            logger.info("Configured to use GPU.")

            # Conditionally set XLA performance flags
            if xla_performance_flags:
                os.environ["XLA_FLAGS"] = (
                    "--xla_gpu_enable_triton_softmax_fusion=true "
                    "--xla_gpu_triton_gemm_any=True "
                    "--xla_gpu_enable_async_collectives=true "
                    "--xla_gpu_enable_latency_hiding_scheduler=true "
                    "--xla_gpu_enable_highest_priority_async_stream=true "
                )
                logger.info("XLA performance flags enabled.")
            else:
                logger.info("XLA performance flags not enabled.")

            # Limit memory usage to a fraction
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
            logger.info(f"Memory fraction set to {mem_fraction}.")

            # Select specified GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            logger.info(f"CUDA_VISIBLE_DEVICES set to {gpu_index}.")

            # Set JAX_CHECK_TRACER_LEAKS based on parameter
            os.environ["JAX_CHECK_TRACER_LEAKS"] = "true" if check_tracer_leaks else "false"
            logger.info(
                f"JAX_CHECK_TRACER_LEAKS set to {'true' if check_tracer_leaks else 'false'}."
            )

            # Get and print backend platform
            backend = xla_bridge.get_backend().platform
            logger.info(f"Backend in use: {backend}")

            # Get and print GPU devices
            gpus = jax.devices("gpu")
            if gpus:
                logger.info("GPUs used:")
                for gpu in gpus:
                    logger.info(f"  - {gpu}")
            else:
                logger.info("No GPUs available.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


# Custom MinMax scaling using JAX
def min_max_scale(X, feature_range=(-jnp.pi / 2, jnp.pi / 2)):
    min_val = jnp.min(X, axis=0, keepdims=True)
    max_val = jnp.max(X, axis=0, keepdims=True)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    X_scaled = scale * (X - min_val) + feature_range[0]
    return X_scaled


# Example usage
if __name__ == "__main__":
    jax_setup(use_cpu=False, mem_fraction=0.5, gpu_index=0, check_tracer_leaks=True)
