import logging
import functools
import os
from datetime import datetime

def setup_logger():
    """
    Configures the logger to write to both a timestamped file and the console.
    """
    # Create a 'logs' directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Generate a unique log filename with the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(logs_dir, f'run_{current_time}.log')

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level of messages to handle

    # Prevent adding duplicate handlers if the function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def log_function_call(func):
    """
    A decorator that logs the start, end, and exceptions of a function call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract filename from the function's metadata
        filename = os.path.basename(func.__globals__.get('__file__', 'unknown_file'))
        logging.info(f"[DEBUGGER] ==> Starting execution of '{func.__name__}' from '{filename}'...")
        try:
            result = func(*args, **kwargs)
            logging.info(f"[DEBUGGER] <== Finished execution of '{func.__name__}' from '{filename}'.")
            return result
        except Exception as e:
            logging.error(f"[DEBUGGER] !!! Exception in '{func.__name__}' from '{filename}': {e}", exc_info=True)
            # Re-raise the exception after logging
            raise
    return wrapper

