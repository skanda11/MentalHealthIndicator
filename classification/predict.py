import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import logging
from tqdm import tqdm
from utils.helpers import load_data, save_data
from utils.debug_logger import log_function_call
from utils.monitor_usage import monitor_resources
import math
import os
import glob
import multiprocessing as mp
from datetime import datetime

@log_function_call
def main(config):
    """
    Main function for local prediction with a highly memory-efficient chunking process
    and integrated resource monitoring.
    """
    # --- Setup Monitoring ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    monitor_file = os.path.join(log_dir, f"usage_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Start the monitoring process
    monitor_process = mp.Process(target=monitor_resources, args=(monitor_file,), daemon=True)
    monitor_process.start()
    logging.info(f"Started background resource monitor. Log will be saved to {monitor_file}")

    try:
        # --- Device Configuration ---
        device = torch.device("cpu")
        logging.info(f"Using device: {device} for optimized local prediction")

        # --- Load Model and Tokenizer ---
        try:
            logging.info(f"Loading model from {config['model']['classifier_path']}")
            model = DistilBertForSequenceClassification.from_pretrained(config['model']['classifier_path'])
            tokenizer = DistilBertTokenizer.from_pretrained(config['model']['tokenizer_path'])
            model.eval()
        except Exception as e:
            logging.error(f"Error loading model/tokenizer: {e}")
            return

        # --- Apply Dynamic Quantization ---
        logging.info("Applying dynamic quantization to the model for faster CPU performance...")
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        logging.info("Model quantization complete.")

        # --- Setup for Chunk Processing ---
        processed_data_path = config['data']['processed_data_path']
        predictions_path = config['data']['predictions_path']
        temp_dir = "data/predictions/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        logging.info(f"Cleaning up old temporary files from {temp_dir} if they exist.")
        for f in glob.glob(os.path.join(temp_dir, "*.csv")):
            os.remove(f)

        chunk_size = 10000  # Process 10,000 rows at a time
        
        try:
            total_rows = sum(1 for row in open(processed_data_path, 'r', encoding='utf-8')) - 1
            num_chunks = math.ceil(total_rows / chunk_size)
        except FileNotFoundError:
            logging.error(f"Prediction data file not found at {processed_data_path}.")
            return

        logging.info(f"Starting local prediction on {total_rows} records...")
        df_iterator = pd.read_csv(processed_data_path, chunksize=chunk_size, iterator=True)
        rows_processed = 0

        # --- Process, Predict, and Save Chunks ---
        with torch.no_grad():
            for i, chunk_df in enumerate(tqdm(df_iterator, total=num_chunks, desc="Processing Chunks")):
                chunk_start_row = rows_processed
                chunk_end_row = chunk_start_row + len(chunk_df) - 1
                logging.info(f"Processing chunk {i + 1}/{num_chunks} (rows {chunk_start_row} to {chunk_end_row})...")
                
                texts = chunk_df['text'].astype(str).tolist()
                
                mini_batch_size = config['prediction']['batch_size']
                chunk_predictions = []
                for j in range(0, len(texts), mini_batch_size):
                    mini_batch_start_row = chunk_start_row + j
                    mini_batch_end_row = min(mini_batch_start_row + mini_batch_size - 1, chunk_end_row)
                    logging.debug(f"  - Tokenizing and predicting mini-batch for rows {mini_batch_start_row} to {mini_batch_end_row}.")

                    mini_batch = texts[j:j + mini_batch_size]
                    encodings = tokenizer(mini_batch, truncation=True, padding=True, return_tensors="pt")
                    
                    input_ids = encodings['input_ids'].to(device)
                    attention_mask = encodings['attention_mask'].to(device)

                    outputs = quantized_model(input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    chunk_predictions.extend(predictions.cpu().numpy())

                # Add predictions to the chunk and save it immediately
                chunk_df['prediction'] = chunk_predictions
                temp_file_path = os.path.join(temp_dir, f"chunk_{i}.csv")
                chunk_df.to_csv(temp_file_path, index=False)
                logging.info(f"  -> Successfully saved predictions for this chunk to {temp_file_path}")
                rows_processed += len(chunk_df)

        # --- Combine Temporary Files ---
        logging.info(f"All chunks processed. Combining {len(glob.glob(os.path.join(temp_dir, '*.csv')))} temporary files into final prediction file...")
        temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.csv"))) # Sort to ensure order
        
        df_list = (pd.read_csv(f) for f in temp_files)
        final_df = pd.concat(df_list, ignore_index=True)
        
        save_data(final_df, predictions_path)
        
        # --- Clean Up Temporary Files ---
        logging.info("Cleaning up temporary files.")
        for f in temp_files:
            os.remove(f)
        os.rmdir(temp_dir)
        logging.info("Prediction process complete.")
        
    finally:
        # --- Stop Monitoring ---
        if monitor_process.is_alive():
            monitor_process.terminate()
            monitor_process.join()
            logging.info("Stopped background resource monitor.")

if __name__ == '__main__':
    from utils.helpers import load_config
    config = load_config()
    if config:
        main(config)

