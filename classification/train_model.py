import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  # <-- FIXED IMPORT
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification
)
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from utils.helpers import load_data, save_data, load_config, create_directories
from utils.debug_logger import log_function_call
import os
from datetime import datetime
import torch.cuda.amp as amp # Import Automatic Mixed Precision

def setup_training_log():
    """Sets up a CSV log file for training metrics."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_metrics.csv")
    
    # Create the file and write the header if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,epoch,avg_train_loss,avg_val_loss\n")
            
    return log_file

class SuicidalTextDataset(torch.utils.data.Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure all values are tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Ensure labels are long type
        return item

    def __len__(self):
        return len(self.labels)

# Dynamically get model and tokenizer classes based on name
def get_model_and_tokenizer(model_name):
    if 'albert' in model_name.lower():
        return AlbertForSequenceClassification, AlbertTokenizer
    elif 'mobilebert' in model_name.lower():
         return MobileBertForSequenceClassification, MobileBertTokenizer
    else: # Default or DistilBERT
        return DistilBertForSequenceClassification, DistilBertTokenizer

@log_function_call
def main(config):
    """
    Main function to fine-tune the model with early stopping and mixed precision.
    """
    # --- Setup Training Log ---
    training_log_file = setup_training_log()

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    use_amp = torch.cuda.is_available()
    if use_amp:
        logging.info("CUDA available, enabling Automatic Mixed Precision (AMP).")
    else:
        logging.warning("CUDA not available, AMP disabled. Training will run on CPU.")

    # --- Load and Split Data ---
    df = load_data(config['data']['processed_data_path'])
    if df is None:
        return

    # Handle subsetting for faster testing runs
    subset_size = config['training'].get('training_subset_size', -1)
    if subset_size > 0 and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=config['preprocessing']['random_state'])
        logging.info(f"Using a subset of {len(df)} training examples.")
    else:
        logging.info(f"Using full dataset of {len(df)} examples.")

    
    # Ensure text column is string
    df['text'] = df['text'].astype(str)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state']
    )

    # --- Tokenization ---
    base_model_name = config['model']['base_model']
    logging.info(f"Loading tokenizer: {base_model_name}")
    
    model_class, tokenizer_class = get_model_and_tokenizer(base_model_name)
    tokenizer = tokenizer_class.from_pretrained(base_model_name)
        
    logging.info("Tokenizing training and testing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])

    # --- Create Datasets and Dataloaders ---
    train_dataset = SuicidalTextDataset(train_encodings, train_labels)
    test_dataset = SuicidalTextDataset(test_encodings, test_labels)

    num_workers = 2 if device.type == 'cuda' else 0
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if use_amp else False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=num_workers, pin_memory=True if use_amp else False)

    # --- Model and Optimizer ---
    logging.info(f"Loading base model: {base_model_name}")
    model = model_class.from_pretrained(base_model_name, num_labels=2)
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- AMP GradScaler ---
    scaler = amp.GradScaler(enabled=use_amp)

    # --- Early Stopping Initialization ---
    patience = int(config['training']['early_stopping']['patience'])
    min_delta = float(config['training']['early_stopping']['min_delta'])
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # --- Training Loop ---
    num_epochs = config['training']['epochs']
    logging.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        logging.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        # Wrap the dataloader with tqdm for a progress bar
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for i, batch in enumerate(train_progress_bar):
            optimizer.zero_grad()
            
            # --- NEW: Detailed logging per batch ---
            if (i + 1) % 100 == 0: # Log every 100 batches
                logging.info(f"Epoch {epoch+1}, processing batch {i+1}/{len(train_dataloader)}")
            
            try:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                # Cast operations to mixed precision
                with amp.autocast(enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                if loss is None:
                    logging.warning(f"Loss is None for batch {i+1}. Skipping batch.")
                    continue

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                # Update tqdm description with current loss
                train_progress_bar.set_postfix({'loss': loss.item()})
            
            except Exception as e:
                logging.error(f"Error during training batch {i+1} in epoch {epoch+1}: {e}")
                # Log the problematic batch's text for review
                try:
                    # Get the original text indices for this batch
                    start_idx = i * config['training']['batch_size']
                    end_idx = start_idx + len(batch['input_ids'])
                    problem_texts = train_texts[start_idx:end_idx]
                    logging.error(f"Problematic texts (first 5): {problem_texts[:5]}")
                except Exception as log_e:
                    logging.error(f"Could not retrieve problematic text: {log_e}")
                
                # Continue to the next batch instead of crashing
                # You might want to 'raise e' here if you prefer to stop on error
                continue 

        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)
        with torch.no_grad():
            for batch in val_progress_bar:
                try:
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    with amp.autocast(enabled=use_amp):
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    if loss is not None:
                        total_val_loss += loss.item()
                        val_progress_bar.set_postfix({'val_loss': loss.item()})
                except Exception as e:
                    logging.error(f"Error during validation batch: {e}")
                    continue # Skip problematic validation batch

        avg_val_loss = total_val_loss / len(test_dataloader) if len(test_dataloader) > 0 else 0
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # -- Log metrics to CSV file --
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(training_log_file, "a") as f:
                f.write(f"{timestamp},{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
        except Exception as e:
            logging.error(f"Could not write to training log file: {e}")

        # -- Early Stopping Check --
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save the best model state
            logging.info("Validation loss improved. Saving best model state.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s). Patience: {patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Save the Best Model ---
    logging.info("Training complete. Saving model...")
    if best_model_state:
        logging.info("Loading best model state before saving.")
        model.load_state_dict(best_model_state)
    else:
        logging.info("No best model state found (or patience=0). Saving the final model state.")
        
    try:
        model.save_pretrained(config['model']['classifier_path'])
        tokenizer.save_pretrained(config['model']['tokenizer_path'])
        logging.info(f"Model and tokenizer saved to {config['model']['classifier_path']}")
    except Exception as e:
        logging.error(f"Error saving model/tokenizer: {e}")


if __name__ == '__main__':
    from utils.helpers import load_config, create_directories
    config = load_config()
    if config:
        create_directories(config)
        main(config)

