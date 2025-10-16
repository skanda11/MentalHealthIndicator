import yaml
import os
import pandas as pd
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def create_directories(config):
    """Creates necessary directories specified in the config file if they don't exist."""
    try:
        # Create parent directories for data, models, etc.
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        Path(config['model']['tokenizer_path']).mkdir(parents=True, exist_ok=True)
        Path(config['model']['classifier_path']).mkdir(parents=True, exist_ok=True)
        print("All required directories are present or have been created.")
    except Exception as e:
        print(f"Error creating directories: {e}")

def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def save_data(df, file_path):
    """Saves a pandas DataFrame to a CSV file."""
    try:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
