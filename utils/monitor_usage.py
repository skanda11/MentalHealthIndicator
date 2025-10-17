import csv
import time
import psutil
from datetime import datetime
import os
import logging

# Set up basic logging for the monitor script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
    logging.info("NVIDIA GPU monitoring enabled.")
except (ImportError, pynvml.NVMLError):
    GPU_ENABLED = False
    logging.warning("pynvml library not found or NVIDIA GPU not detected. GPU monitoring will be disabled.")

def monitor_resources(output_file, interval=1):
    """
    Monitors system resources (CPU, RAM, GPU) at a given interval and logs to a CSV.
    
    Args:
        output_file (str): The path to the CSV file where logs will be saved.
        interval (int): The monitoring interval in seconds.
    """
    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the header to the CSV file
    header = [
        "timestamp", "cpu_percent", "ram_percent", "ram_used_gb",
        "gpu_percent", "gpu_memory_percent", "gpu_memory_used_gb"
    ]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    logging.info(f"Starting resource monitoring. Logging to {output_file} every {interval} second(s).")
    
    try:
        while True:
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # CPU and RAM Usage
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = round(ram.used / (1024**3), 2)
            
            # GPU Usage (if enabled)
            gpu_percent = 'N/A'
            gpu_memory_percent = 'N/A'
            gpu_memory_used_gb = 'N/A'
            
            if GPU_ENABLED:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_percent = round((gpu_mem.used / gpu_mem.total) * 100, 2)
                    gpu_memory_used_gb = round(gpu_mem.used / (1024**3), 2)
                except pynvml.NVMLError as e:
                    logging.error(f"Could not retrieve GPU stats: {e}")
                    # In case of error, we keep logging other stats
            
            # Write data to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, cpu_percent, ram_percent, ram_used_gb,
                    gpu_percent, gpu_memory_percent, gpu_memory_used_gb
                ])
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("Resource monitoring stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred in the monitoring loop: {e}")
    finally:
        if GPU_ENABLED:
            pynvml.nvmlShutdown()

if __name__ == '__main__':
    # This allows the script to be run directly for testing
    output_csv = os.path.join("logs", "system_usage.csv")
    monitor_resources(output_csv)
