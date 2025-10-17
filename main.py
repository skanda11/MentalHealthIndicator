import argparse
import sys
import logging
import os
from utils.helpers import load_config, create_directories
from preprocessing.preprocess import main as preprocess_main
from classification.train_model import main as train_main
from classification.predict import main as predict_main
from classification.predict_gemini import main as predict_gemini_main
from temporal_analysis.analyze_trends import main as analyze_main

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the pipeline execution based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Mental Health Indicator Analysis Pipeline")
    parser.add_argument(
        'steps',
        nargs='+',
        choices=['preprocess', 'train', 'predict', 'predict_gemini', 'analyze', 'dashboard', 'all'],
        help="Pipeline step(s) to run. Use 'all' to run the entire pipeline."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config()
    if not config:
        logging.error("Could not load configuration. Exiting.")
        return
        
    create_directories(config)
    
    # --- Execute Pipeline Steps ---
    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['preprocess', 'train', 'predict', 'analyze']

    if 'preprocess' in steps_to_run:
        logging.info("--- Starting Step: Data Preprocessing ---")
        preprocess_main(config)
        logging.info("--- Finished Step: Data Preprocessing ---")

    if 'train' in steps_to_run:
        logging.info("--- Starting Step: Model Training ---")
        train_main(config)
        logging.info("--- Finished Step: Model Training ---")

    if 'predict' in steps_to_run:
        logging.info("--- Starting Step: Prediction ---")
        predict_main(config)
        logging.info("--- Completed Step: Prediction ---")

    if 'predict_gemini' in steps_to_run:
        logging.info("--- Starting Step: Prediction with Gemini ---")
        predict_gemini_main(config)
        logging.info("--- Completed Step: Prediction with Gemini ---")

    if 'analyze' in steps_to_run:
        logging.info("--- Starting Step: Temporal Analysis ---")
        analyze_main(config)
        logging.info("--- Finished Step: Temporal Analysis ---")

    if 'dashboard' in steps_to_run:
        logging.info("--- Starting Step: Launching Dashboard ---")
        os.system("streamlit run dashboard/dashboard.py")

if __name__ == '__main__':
    main()

