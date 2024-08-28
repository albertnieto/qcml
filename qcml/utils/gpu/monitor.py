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

import threading
import time
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetName,
    nvmlDeviceGetCount,
    nvmlShutdown,
    nvmlSystemGetProcessName,
)
from IPython.display import display, clear_output, DisplayHandle, Pretty
import jax
from jax.lib import xla_bridge


def monitor_gpu(interval=1, display_handle=None, show_jax=False):
    """
    Monitors GPU usage and prints information about memory usage,
    utilization, temperature, and running processes, along with optional JAX execution profiling information.

    Parameters:
    interval (int): The interval (in seconds) at which to refresh the GPU information.
    display_handle (DisplayHandle): The handle to the display object that controls output.
    show_jax (bool): Whether to include JAX profiling information in the output.
    """
    nvmlInit()
    try:
        device_count = nvmlDeviceGetCount()
        while True:
            output = []
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                utilization = nvmlDeviceGetUtilizationRates(handle)
                temperature = nvmlDeviceGetTemperature(handle, 0)
                processes = nvmlDeviceGetComputeRunningProcesses(handle)

                output.append(f"GPU ID: {i} - {name}")
                output.append(f"  Total Memory: {mem_info.total / (1024 ** 2):.2f} MB")
                output.append(f"  Used Memory: {mem_info.used / (1024 ** 2):.2f} MB")
                output.append(f"  Free Memory: {mem_info.free / (1024 ** 2):.2f} MB")
                output.append(f"  GPU Utilization: {utilization.gpu}%")
                output.append(f"  Temperature: {temperature} °C")
                output.append("  Processes:")

                for process in processes:
                    process_name = "Unknown"
                    try:
                        process_name = nvmlSystemGetProcessName(process.pid).decode(
                            "utf-8"
                        )
                    except Exception as e:
                        process_name = "Process name unavailable"

                    output.append(f"    PID: {process.pid}")
                    output.append(f"    Process Name: {process_name}")
                    output.append(
                        f"    Used GPU Memory: {process.usedGpuMemory / (1024 ** 2):.2f} MB"
                    )
                    output.append(
                        f"    Process Type: {process.type}"
                        if hasattr(process, "type")
                        else ""
                    )
                    output.append("-" * 20)

                output.append("-" * 40)

            # Conditionally include JAX profiling information
            if show_jax:
                try:
                    jax_device = jax.devices("gpu")[0]
                    output.append(f"JAX Device: {jax_device}")
                    output.append(f"JAX Backend: {xla_bridge.get_backend().platform}")
                    output.append(f"JAX Local Device Count: {jax.local_device_count()}")
                    output.append(
                        f"JAX Total Allocated Buffers: {len(jax_device.live_buffers())}"
                    )
                except Exception as e:
                    output.append(f"JAX Profiling Error: {str(e)}")

                output.append("-" * 40)

            # Join the output lines
            pretty_output = Pretty("\n".join(output))
            if display_handle:
                display_handle.update(pretty_output)
            else:
                display(pretty_output)
            time.sleep(interval)
    finally:
        nvmlShutdown()


def start_gpu_monitor(interval=1, show_jax=False):
    """
    Starts the GPU monitor in a separate thread so that it doesn't block
    other code execution. Uses a DisplayHandle to manage output updates.

    Parameters:
    interval (int): The interval (in seconds) at which to refresh the GPU information.
    show_jax (bool): Whether to include JAX profiling information in the output.
    """
    display_handle = DisplayHandle()
    display_handle.display(Pretty("Initializing GPU monitor..."))

    monitor_thread = threading.Thread(
        target=monitor_gpu, args=(interval, display_handle, show_jax), daemon=True
    )
    monitor_thread.start()


if __name__ == "__main__":
    # Example: Start monitoring with a refresh interval of 1 second
    start_gpu_monitor(interval=1, show_jax=True)

    # Simulate doing other work in the main thread
    print("GPU monitor started. Doing other work in the main thread.")
    for i in range(10):
        print(f"Main thread work iteration {i + 1}")
        time.sleep(2)
