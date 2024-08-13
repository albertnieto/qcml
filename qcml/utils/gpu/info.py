import jax
import logging

logger = logging.getLogger(__name__)

def get_gpu_info():
    try:
        gpus = jax.devices("gpu")
        num_gpus = len(gpus)
        gpu_names = [gpu.device_kind for gpu in gpus]
        return num_gpus, gpu_names
    except Exception as e:
        logger.error(f"Error retrieving GPU information: {e}")
        return 0, []