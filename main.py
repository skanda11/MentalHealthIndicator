import argparse
import sys
import logging
import subprocess  # <-- Import the subprocess module
from utils.helpers import load_config, create_directories
from preprocessing.preprocess import main as preprocess_main
from tokenization.tokenize_data import main as tokenize_main 
from classification.train_model import main as train_main
from classification.predict import main as predict_main
from classification.evaluate_model import main as evaluate_main
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
        choices=['preprocess', 'tokenize', 'train', 'predict', 'evaluate', 'analyze', 'all',
                 'dashboard', 'gemini_chat', 'hybrid_chat'], # <-- ADDED NEW APP CHOICES
        help="Pipeline step(s) to run. 'all' runs the data pipeline. 'dashboard', 'gemini_chat', and 'hybrid_chat' launch the Streamlit apps."
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
        steps_to_run = ['preprocess', 'tokenize', 'train', 'predict', 'evaluate', 'analyze']

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

    if 'evaluate' in steps_to_run:
        logging.info("--- Starting Step: Model Evaluation ---")
        evaluate_main(config)
        logging.info("--- Completed Step: Model Evaluation ---")

    if 'analyze' in steps_to_run:
        logging.info("--- Starting Step: Temporal Analysis ---")
        analyze_main(config)
        logging.info("--- Completed Step: Temporal Analysis ---")

    # --- ADDED: Launch Streamlit Apps ---

    if 'dashboard' in steps_to_run:
        logging.info("--- Launching Main Dashboard ---")
        logging.info("Access the app in your browser (usually at http://localhost:8501)")
        try:
            # Note: This will block the terminal until you close the app
            subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])
        except FileNotFoundError:
            logging.error("Could not find 'streamlit'. Please ensure it's installed: pip install streamlit")

    if 'gemini_chat' in steps_to_run:
        logging.info("--- Launching Gemini Chatbot (API only) ---")
        logging.info("Access the app in your browser (usually at http://localhost:8501)")
        try:
            subprocess.run(["streamlit", "run", "chatbot.py"])
        except FileNotFoundError:
            logging.error("Could not find 'streamlit'. Please ensure it's installed: pip install streamlit")

    if 'hybrid_chat' in steps_to_run:
        logging.info("--- Launching Hybrid Chatbot (Local Model + API) ---")
        logging.info("Access the app in your browser (usually at http://localhost:8501)")
        try:
            # Assumes hybrid_chatbot.py is in the root project directory
            subprocess.run(["streamlit", "run", "hybrid_chatbot.py"]) 
        except FileNotFoundError:
            logging.error("Could not find 'streamlit'. Please ensure it's installed: pip install streamlit")

    logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()