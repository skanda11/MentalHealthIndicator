import torch
from torch.utils.data import DataLoader, TensorDataset # <-- Import TensorDataset
from torch.optim import AdamW
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

# Note: The SuicidalTextDataset class is no longer needed
# as we will use PyTorch's built-in TensorDataset

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

    # --- Load Pre-Tokenized Data ---
    tokenized_data_path = config['data']['tokenized_data_path']
    try:
        logging.info(f"Loading pre-tokenized data from {tokenized_data_path}...")
        tokenized_data = torch.load(tokenized_data_path)
        all_input_ids = tokenized_data['input_ids']
        all_attention_masks = tokenized_data['attention_mask']
        all_labels = tokenized_data['labels']
        logging.info("Tokenized data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Tokenized data file '{tokenized_data_path}' not found.")
        logging.error("Please run the 'tokenize' step first. You can run: python tokenization/tokenize_data.py")
        return
    except Exception as e:
        logging.error(f"Error loading tokenized data: {e}")
        return
    
    # --- Handle Subsetting ---
    subset_size = config['training'].get('training_subset_size', -1)
    test_subset_size = config['training'].get('testing_subset_size', 500) 

    if subset_size > 0 and (subset_size + test_subset_size) < len(all_labels):
        logging.info(f"Using a subset of {subset_size} for training and {test_subset_size} for testing.")
        
        # Calculate total subset size and test proportion
        total_subset = subset_size + test_subset_size
        test_proportion = test_subset_size / float(total_subset)

        # Stratified split to get a representative subset
        temp_indices = np.arange(len(all_labels))
        _, subset_indices = train_test_split(
            temp_indices,
            test_size=total_subset,
            random_state=config['preprocessing']['random_state'],
            stratify=all_labels
        )
        
        subset_input_ids = all_input_ids[subset_indices]
        subset_attention_masks = all_attention_masks[subset_indices]
        subset_labels = all_labels[subset_indices]

        # Now, split this subset into train and test
        train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
            subset_input_ids,
            subset_attention_masks,
            subset_labels,
            test_size=test_proportion,
            random_state=config['preprocessing']['random_state'],
            stratify=subset_labels
        )
    
    else:
        logging.info(f"Using full dataset of {len(all_labels)} examples.")
        # Split the full dataset into train and test
        train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
            all_input_ids,
            all_attention_masks,
            all_labels,
            test_size=config['preprocessing']['test_size'],
            random_state=config['preprocessing']['random_state'],
            stratify=all_labels
        )

    # --- Create Datasets and Dataloaders ---
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    num_workers = 2 if device.type == 'cuda' else 0
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if use_amp else False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=num_workers, pin_memory=True if use_amp else False)

    # --- Model and Optimizer ---
    base_model_name = config['model']['base_model']
    logging.info(f"Loading tokenizer: {base_model_name}")
    
    model_class, tokenizer_class = get_model_and_tokenizer(base_model_name)
    # We still need the tokenizer for saving later
    tokenizer = tokenizer_class.from_pretrained(base_model_name)
        
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
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for i, batch in enumerate(train_progress_bar):
            optimizer.zero_grad()
            
            if (i + 1) % 100 == 0: # Log every 100 batches
                logging.info(f"Epoch {epoch+1}, processing batch {i+1}/{len(train_dataloader)}")
            
            try:
                # Unpack the batch from TensorDataset
                input_ids, attention_mask, labels = [b.to(device, non_blocking=True) for b in batch]

                # Cast operations to mixed precision
                with amp.autocast(enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                if loss is None:
                    logging.warning(f"Loss is None for batch {i+1}. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                train_progress_bar.set_postfix({'loss': loss.item()})
            
            except Exception as e:
                logging.error(f"Error during training batch {i+1} in epoch {epoch+1}: {e}")
                # Skipping detailed text logging as it's complex with pre-tokenized data
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
                    # Unpack the batch from TensorDataset
                    input_ids, attention_mask, labels = [b.to(device, non_blocking=True) for b in batch]

                    with amp.autocast(enabled=use_amp):
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    if loss is not None:
                        total_val_loss += loss.item()
                        val_progress_bar.set_postfix({'val_loss': loss.item()})
                except Exception as e:
                    logging.error(f"Error during validation batch: {e}")
                    continue

        # *** THIS IS THE FIX FOR THE NameError ***
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