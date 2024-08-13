import threading
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetName, nvmlDeviceGetCount, nvmlShutdown
from IPython.display import display, clear_output, DisplayHandle, Pretty

def monitor_gpu(interval=1, display_handle=None):
    """
    Monitors GPU usage and prints information about memory usage,
    utilization, temperature, and running processes.

    Parameters:
    interval (int): The interval (in seconds) at which to refresh the GPU information.
    display_handle (DisplayHandle): The handle to the display object that controls output.
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
                    output.append(f"    PID: {process.pid}, Used Memory: {process.usedGpuMemory / (1024 ** 2):.2f} MB")
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

def start_gpu_monitor(interval=1):
    """
    Starts the GPU monitor in a separate thread so that it doesn't block
    other code execution. Uses a DisplayHandle to manage output updates.

    Parameters:
    interval (int): The interval (in seconds) at which to refresh the GPU information.
    """
    display_handle = DisplayHandle()
    display_handle.display(Pretty("Initializing GPU monitor..."))
    
    monitor_thread = threading.Thread(target=monitor_gpu, args=(interval, display_handle), daemon=True)
    monitor_thread.start()

if __name__ == "__main__":
    # Example: Start monitoring with a refresh interval of 1 second
    start_gpu_monitor(interval=1)
    
    # Simulate doing other work in the main thread
    print("GPU monitor started. Doing other work in the main thread.")
    for i in range(10):
        print(f"Main thread work iteration {i + 1}")
        time.sleep(2)
