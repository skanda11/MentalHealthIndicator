import pandas as pd
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    """
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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    logging.info("--- Model Evaluation Report ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info("---------------------------------")
    
    # --- Detailed Classification Report ---
    report = classification_report(y_true, y_pred, target_names=['Non-Suicide (0)', 'Suicide Risk (1)'])
    logging.info("Classification Report:\n" + report)
    
    # --- Confusion Matrix ---
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
        # plt.show() # This would open the plot, but saving is better for a script
    except Exception as e:
        logging.warning(f"Could not generate confusion matrix plot. Error: {e}")
        logging.warning("Ensure you have 'matplotlib' and 'seaborn' installed: pip install matplotlib seaborn")

if __name__ == '__main__':
    config = load_config()
    if config:
        main(config)