import argparse
import sys
import logging
from utils.helpers import load_config, create_directories
from preprocessing.preprocess import main as preprocess_main
# We are not using the separate tokenizer script in this version
# from tokenization.tokenize_data import main as tokenize_main 
from classification.train_model import main as train_main
from classification.predict import main as predict_main
from temporal_analysis.analyze_trends import main as analyze_main
from utils.debug_logger import setup_logger # Import the logger setup

# --- Setup Logging ---
setup_logger() # Call the function to configure file logging

def main():
    """
    Main function to orchestrate the pipeline execution based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Mental Health Indicator Analysis Pipeline")
    parser.add_argument(
        'steps',
        nargs='+',
        choices=['preprocess', 'train', 'predict', 'analyze', 'all'], # 'tokenize' is removed
        help="Pipeline step(s) to run. Use 'all' to run the entire pipeline."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        config = load_config()
        if not config:
            logging.error("Configuration file 'config.yaml' could not be loaded. Exiting.")
            sys.exit(1)
        logging.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logging.error("Configuration file 'config.yaml' not found. Please ensure it's in the project root.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # --- Create Necessary Directories ---
    create_directories(config)

    # --- Execute Pipeline Steps ---
    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['preprocess', 'train', 'predict', 'analyze']

    if 'preprocess' in steps_to_run:
        logging.info("--- Starting Step: Data Preprocessing ---")
        preprocess_main(config)
        logging.info("--- Completed Step: Data Preprocessing ---")

    # 'tokenize' step is removed

    if 'train' in steps_to_run:
        logging.info("--- Starting Step: Model Training ---")
        train_main(config)
        logging.info("--- Completed Step: Model Training ---")

    if 'predict' in steps_to_run:
        logging.info("--- Starting Step: Prediction ---")
        predict_main(config)
        logging.info("--- Completed Step: Prediction ---")

    if 'analyze' in steps_to_run:
        logging.info("--- Starting Step: Temporal Analysis ---")
        analyze_main(config)
        logging.info("--- Completed Step: Temporal Analysis ---")

    logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()

# Note: The tokenization step has been removed as per the updated requirements.