import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import logging
from tqdm import tqdm
from utils.helpers import load_data, save_data
from utils.debug_logger import log_function_call

@log_function_call
def main(config):
    """
    Main function to generate predictions using the fine-tuned model.
    """
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    try:
        logging.info(f"Loading model from {config['model']['classifier_path']}")
        model = DistilBertForSequenceClassification.from_pretrained(config['model']['classifier_path'])
        tokenizer = DistilBertTokenizer.from_pretrained(config['model']['tokenizer_path'])
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        logging.error("Please ensure the model has been trained and paths are correct in config.yaml.")
        return

    # --- Load Processed Data ---
    df = load_data(config['data']['processed_data_path'])
    if df is None:
        return
        
    # Ensure the text column is a list of strings to prevent tokenizer errors
    # This converts any non-string data (like NaN) to an empty string
    texts = df['text'].astype(str).tolist()

    logging.info("Tokenizing data for prediction...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'], return_tensors="pt")

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
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
    df['prediction'] = all_predictions
    save_data(df, config['data']['predictions_path'])
    logging.info(f"Predictions saved to {config['data']['predictions_path']}")

if __name__ == '__main__':
    from utils.helpers import load_config
    config = load_config()
    if config:
        main(config)

