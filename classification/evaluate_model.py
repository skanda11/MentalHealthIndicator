import pandas as pd
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime  # <-- Added for timestamping

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------

from utils.helpers import load_config, load_data
from utils.debug_logger import setup_logger, log_function_call

# Setup logger
setup_logger()

@log_function_call
def main(config):
    """
    Loads predictions and evaluates the model's performance.
    Saves the results to logs/evaluation_results.txt
    """
    # --- Define output file path ---
    results_file_path = os.path.join('logs', 'evaluation_results.txt')

    # --- Load Predictions Data ---
    predictions_path = config['data']['predictions_path']
    logging.info(f"Loading predictions from {predictions_path}...")
    
    df = load_data(predictions_path)
    if df is None:
        logging.error(f"Could not load predictions file. Did you run the 'predict' step?")
        return
        
    if 'label' not in df.columns or 'prediction' not in df.columns:
        logging.error("Predictions file is missing 'label' or 'prediction' columns.")
        return

    y_true = df['label']
    y_pred = df['prediction']
    
    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # --- Log to console (as before) ---
    logging.info("--- Model Evaluation Report ---")
    logging.info(f"Accuracy:  {accuracy:.4f} (Overall correctness)")
    logging.info(f"Precision: {precision:.4f} (Of positive predictions, how many were correct?)")
    logging.info(f"Recall:    {recall:.4f} (Of all actual positives, how many were found?)")
    logging.info(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
    logging.info("---------------------------------")
    
    # --- Detailed Classification Report ---
    report = classification_report(y_true, y_pred, target_names=['Non-Suicide (0)', 'Suicide Risk (1)'], zero_division=0)
    logging.info("Classification Report:\n" + report)

    # --- NEW: Save results to evaluation_results.txt ---
    try:
        with open(results_file_path, 'w') as f:
            f.write(f"Model Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*30 + "\n")
            f.write(f"Model: {config['model']['base_model']}\n")
            f.write(f"Prediction File: {predictions_path}\n")
            f.write(f"Total Test Samples: {len(y_true)}\n")
            f.write("="*30 + "\n\n")
            
            f.write("--- Summary Metrics ---\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")
            
            f.write("--- Detailed Classification Report ---\n")
            f.write(report)
        
        logging.info(f"Evaluation results successfully saved to {results_file_path}")

    except Exception as e:
        logging.error(f"Failed to write results to {results_file_path}. Error: {e}")
    
    # --- Confusion Matrix (as before) ---
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Non-Suicide', 'Predicted Suicide Risk'],
                    yticklabels=['Actual Non-Suicide', 'Actual Suicide Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        # Save the confusion matrix to the logs folder
        cm_path = os.path.join('logs', 'confusion_matrix.png')
        plt.savefig(cm_path)
        logging.info(f"Confusion Matrix saved to {cm_path}")
    except Exception as e:
        logging.warning(f"Could not generate confusion matrix plot. Error: {e}")
        logging.warning("Ensure you have 'matplotlib' and 'seaborn' installed: pip install matplotlib seaborn")

if __name__ == '__main__':
    config = load_config()
    if config:
        main(config)