import torch
from transformers import DistilBertTokenizer
import logging
import pandas as pd
from utils.helpers import load_data, load_config
from utils.debug_logger import log_function_call

@log_function_call
def main(config):
    """
    Loads processed data, tokenizes it, and saves the tensors for training and prediction.
    """
    # --- Load Tokenizer ---
    base_model = config['model']['base_model']
    logging.info(f"Loading tokenizer for '{base_model}'...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(base_model)
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return

    # --- Load Processed Data ---
    processed_path = config['data']['processed_data_path']
    df = load_data(processed_path)
    if df is None:
        logging.error("Could not load processed data. Please run the 'preprocess' step first.")
        return

    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    # --- Tokenize Data ---
    logging.info("Tokenizing all text data. This may take a few minutes...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])

    # --- Save Tensors ---
    tokenized_data = {
        'input_ids': torch.tensor(encodings['input_ids']),
        'attention_mask': torch.tensor(encodings['attention_mask']),
        'labels': torch.tensor(labels)
    }
    
    save_path = config['data']['tokenized_data_path']
    torch.save(tokenized_data, save_path)
    logging.info(f"Tokenized data saved successfully to {save_path}")

if __name__ == '__main__':
    config = load_config()
    if config:
        main(config)
