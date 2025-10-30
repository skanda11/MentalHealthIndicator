import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,  # <-- Added
    MobileBertForSequenceClassification # <-- Added
)
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split # <-- Added
from utils.helpers import load_data, save_data, load_config
from utils.debug_logger import log_function_call, setup_logger # <-- Added setup_logger
import os
import sys
import numpy as np # <-- Added

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------

def get_model_class(model_name):
    """Gets the correct model class from the model name in the config."""
    if 'albert' in model_name.lower():
        return AlbertForSequenceClassification
    elif 'mobilebert' in model_name.lower():
        return MobileBertForSequenceClassification
    else: # Default or DistilBERT
        return DistilBertForSequenceClassification

@log_function_call
def main(config):
    """
    Generates predictions using the fine-tuned model on pre-tokenized data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Determine Model Type from Config ---
    base_model_name = config['model']['base_model']
    model_class = get_model_class(base_model_name)
    logging.info(f"Expecting model type: {model_class.__name__}")

    # --- Load Model ---
    try:
        model_path = config['model']['classifier_path']
        logging.info(f"Loading model from {model_path}")
        # Use the determined model class to load the correct model
        model = model_class.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading trained model: {e}")
        logging.error(f"Please ensure the model exists at {model_path} and matches the type '{base_model_name}'.")
        return

    # --- Load Pre-Tokenized Data ---
    tokenized_data_path = config['data']['tokenized_data_path']
    try:
        logging.info(f"Loading pre-tokenized data from {tokenized_data_path}...")
        tokenized_data = torch.load(tokenized_data_path)
        
        all_input_ids = tokenized_data['input_ids']
        all_attention_masks = tokenized_data['attention_mask']
        all_labels = tokenized_data['labels']

    except FileNotFoundError:
        logging.error(f"Tokenized data file not found at {tokenized_data_path}. Please run 'python tokenization/tokenize_data.py' first.")
        return
    except Exception as e:
        logging.error(f"Error loading tokenized data: {e}")
        return
    
    # --- Re-create the Test Split ---
    # We must re-split the *indices* exactly as done in training to get the correct test set
    try:
        subset_size = config['training'].get('training_subset_size', -1)
        test_subset_size = config['training'].get('testing_subset_size', 500)
        
        indices = np.arange(len(all_labels))
        
        if subset_size > 0 and (subset_size + test_subset_size) < len(all_labels):
            total_subset = subset_size + test_subset_size
            test_proportion = test_subset_size / float(total_subset)
            
            _, subset_indices = train_test_split(
                indices,
                test_size=total_subset,
                random_state=config['preprocessing']['random_state'],
                stratify=all_labels
            )
            subset_labels = all_labels[subset_indices]
            
            _, test_indices = train_test_split(
                subset_indices,
                test_size=test_proportion,
                random_state=config['preprocessing']['random_state'],
                stratify=subset_labels
            )
            logging.info(f"Predicting on the test subset of {len(test_indices)} examples.")
            
        else:
            _, test_indices = train_test_split(
                indices,
                test_size=config['preprocessing']['test_size'],
                random_state=config['preprocessing']['random_state'],
                stratify=all_labels
            )
            logging.info(f"Predicting on the full test set of {len(test_indices)} examples.")

        # Select only the test data tensors
        input_ids = all_input_ids[test_indices]
        attention_mask = all_attention_masks[test_indices]
        
        dataset = TensorDataset(input_ids, attention_mask)
        
    except Exception as e:
        logging.error(f"Error splitting tokenized data: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=config['prediction']['batch_size'])

    # --- Prediction Loop ---
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            input_ids_batch, attention_mask_batch = [b.to(device) for b in batch]
            
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # --- Save Predictions ---
    df = load_data(config['data']['processed_data_path'])
    
    if df is not None:
        # Get the original test texts and labels using the same indices
        test_df = df.iloc[test_indices].copy()
        test_df['prediction'] = all_predictions
        
        if len(test_df) == len(all_predictions):
            save_data(test_df[['text', 'label', 'prediction']], config['data']['predictions_path'])
        else:
            logging.error(f"Mismatch between data count ({len(test_df)}) and prediction count ({len(all_predictions)}). Predictions not saved.")
    else:
        logging.error("Could not load processed data to save predictions.")

if __name__ == '__main__':
    setup_logger() # Setup logger if run as main
    config = load_config()
    if config:
        main(config)