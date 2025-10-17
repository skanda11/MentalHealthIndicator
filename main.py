import argparse
import sys
import logging
from utils.helpers import load_config, create_directories
from preprocessing.preprocess import main as preprocess_main
from tokenization.tokenize_data import main as tokenize_main # New import
from classification.train_model import main as train_main
from classification.predict import main as predict_main
from temporal_analysis.analyze_trends import main as analyze_main

# --- Setup Logging ---
# (Logging setup remains the same)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """
    Main function to orchestrate the pipeline execution.
    """
    parser = argparse.ArgumentParser(description="Mental Health Indicator Analysis Pipeline")
    parser.add_argument(
        'steps',
        nargs='+',
        choices=['preprocess', 'tokenize', 'train', 'predict', 'analyze', 'all'], # Added 'tokenize'
        help="Pipeline step(s) to run."
    )
    args = parser.parse_args()

    config = load_config()
    if not config:
        logging.error("Configuration file could not be loaded. Exiting.")
        sys.exit(1)
    
    create_directories(config)

    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['preprocess', 'tokenize', 'train', 'predict', 'analyze']

    if 'preprocess' in steps_to_run:
        logging.info("--- Starting Step: Data Preprocessing ---")
        preprocess_main(config)
        logging.info("--- Completed Step: Data Preprocessing ---")

    if 'tokenize' in steps_to_run:
        logging.info("--- Starting Step: Data Tokenization ---")
        tokenize_main(config)
        logging.info("--- Completed Step: Data Tokenization ---")

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

