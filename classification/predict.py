import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification
import pandas as pd
import logging
from tqdm import tqdm
from utils.helpers import load_data, save_data, load_config
from utils.debug_logger import log_function_call

@log_function_call
def main(config):
    """
    Generates predictions using the fine-tuned model on pre-tokenized data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    try:
        logging.info(f"Loading model from {config['model']['classifier_path']}")
        model = DistilBertForSequenceClassification.from_pretrained(config['model']['classifier_path'])
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading trained model: {e}")
        return

    # --- Load Pre-Tokenized Data ---
    tokenized_data_path = config['data']['tokenized_data_path']
    try:
        logging.info(f"Loading pre-tokenized data from {tokenized_data_path}...")
        tokenized_data = torch.load(tokenized_data_path)
        dataset = TensorDataset(
            tokenized_data['input_ids'],
            tokenized_data['attention_mask']
        )
    except FileNotFoundError:
        logging.error(f"Tokenized data file not found. Please run 'tokenize' step first.")
        return

    dataloader = DataLoader(dataset, batch_size=config['prediction']['batch_size'])

    # --- Prediction Loop ---
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # --- Save Predictions ---
    # Load original processed data to append predictions
    df = load_data(config['data']['processed_data_path'])
    if df is not None and len(df) == len(all_predictions):
        df['prediction'] = all_predictions
        save_data(df, config['data']['predictions_path'])
    else:
        logging.error("Mismatch between data and prediction counts. Predictions not saved.")

if __name__ == '__main__':
    config = load_config()
    if config:
        main(config)

