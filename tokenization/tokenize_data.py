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
    logging.info(f"Loaded {len(texts)} texts and {len(labels)} labels for tokenization.")


    # --- Tokenize Data ---
    logging.info("Starting tokenization of all text data. This may take a few minutes...")
    try:
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])
        logging.info("Tokenization completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during tokenization: {e}")
        return

    # --- Create Tensors ---
    logging.info("Converting tokenized outputs to PyTorch tensors.")
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels_tensor = torch.tensor(labels)

    logging.info(f"Shape of input_ids tensor: {input_ids.shape}")
    logging.info(f"Shape of attention_mask tensor: {attention_mask.shape}")
    logging.info(f"Shape of labels tensor: {labels_tensor.shape}")

    tokenized_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_tensor
    }
    
    # --- Save Tensors ---
    save_path = config['data']['tokenized_data_path']
    logging.info(f"Attempting to save tokenized data to {save_path}...")
    try:
        torch.save(tokenized_data, save_path)
        logging.info(f"Tokenized data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save tokenized data: {e}")


if __name__ == '__main__':
    config = load_config()
    if config:
        main(config)

