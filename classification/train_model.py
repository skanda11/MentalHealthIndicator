import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW # Corrected import
# Import specific classes based on model name later
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
from utils.helpers import load_data
import os
from datetime import datetime
import torch.cuda.amp as amp # Import Automatic Mixed Precision

def setup_training_log():
    """Sets up a CSV log file for training metrics."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,epoch,avg_train_loss,avg_val_loss\n")
    return log_file

class SuicidalTextDataset(torch.utils.data.Dataset):
    """Custom Dataset for PyTorch, loading pre-tokenized data."""
    def __init__(self, encodings, labels):
        # Assuming encodings is a dict like {'input_ids': [...], 'attention_mask': [...]}
        # and labels is a list [...]
        self.encodings = encodings
        self.labels = labels
        # Convert lists of tensors/numbers to tensors directly for efficiency
        self.input_ids = torch.tensor(self.encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(self.encodings['attention_mask'], dtype=torch.long)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels_tensor[idx]
        }
        return item

    def __len__(self):
        # Use the length of one of the encoding lists (e.g., input_ids)
        return len(self.encodings['input_ids'])


# No decorator here, keep main clean
def main(config):
    """
    Main function to fine-tune the model with early stopping and mixed precision.
    Uses pre-tokenized data.
    """
    training_log_file = setup_training_log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Check if CUDA is actually available for AMP
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
        encodings = tokenized_data['encodings'] # Should be dict {'input_ids': list, 'attention_mask': list}
        labels = tokenized_data['labels']       # Should be list
        logging.info("Pre-tokenized data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Tokenized data file not found at {tokenized_data_path}. Please run the 'tokenize' step first.")
        return
    except Exception as e:
        logging.error(f"Error loading tokenized data: {e}")
        return

    # --- Split Data ---
    logging.info("Splitting tokenized data into training and validation sets...")
    indices = list(range(len(labels)))
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=config['preprocessing']['test_size'],
            random_state=config['preprocessing']['random_state'],
            stratify=labels # Stratify helps maintain class balance
        )
    except ValueError as e:
         logging.warning(f"Could not stratify data split (likely too few samples of one class): {e}. Proceeding without stratification.")
         train_indices, val_indices = train_test_split(
            indices,
            test_size=config['preprocessing']['test_size'],
            random_state=config['preprocessing']['random_state']
        )


    # Create NEW dictionaries for train/val encodings based on indices
    train_encodings = {key: [encodings[key][i] for i in train_indices] for key in encodings}
    val_encodings = {key: [encodings[key][i] for i in val_indices] for key in encodings}
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]

    logging.info(f"Data split: {len(train_labels)} training, {len(val_labels)} validation examples.")

    # --- Create Datasets and Dataloaders ---
    train_dataset = SuicidalTextDataset(train_encodings, train_labels)
    val_dataset = SuicidalTextDataset(val_encodings, val_labels)

    # num_workers > 0 can speed up data loading but uses more RAM. Start with 2 if you have enough RAM.
    num_workers = 2 if device.type == 'cuda' else 0 # Use workers only on GPU to avoid CPU overhead issues
    logging.info(f"Using {num_workers} workers for DataLoader.")
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if use_amp else False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=num_workers, pin_memory=True if use_amp else False)

    # --- Model and Optimizer ---
    base_model_name = config['model']['base_model']
    logging.info(f"Loading base model: {base_model_name}")
    try:
        # Dynamically import and load model/tokenizer
        if 'albert' in base_model_name.lower():
            from transformers import AlbertTokenizer, AlbertForSequenceClassification
            tokenizer_class = AlbertTokenizer
            model_class = AlbertForSequenceClassification
        elif 'mobilebert' in base_model_name.lower():
             from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
             tokenizer_class = MobileBertTokenizer
             model_class = MobileBertForSequenceClassification
        else: # Default or DistilBERT
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            tokenizer_class = DistilBertTokenizer
            model_class = DistilBertForSequenceClassification

        # Tokenizer is only needed here to save it later, not for processing
        tokenizer = tokenizer_class.from_pretrained(base_model_name)
        model = model_class.from_pretrained(base_model_name, num_labels=2)

    except Exception as e:
        logging.error(f"Failed to load model/tokenizer {base_model_name}: {e}")
        return

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # --- AMP GradScaler ---
    # Creates a GradScaler once at the beginning of training.
    scaler = amp.GradScaler(enabled=use_amp)

    # --- Early Stopping Initialization ---
    patience = int(config['training'].get('early_stopping_patience', 3))
    min_delta = float(config['training'].get('min_delta', 0.01))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # --- Training Loop ---
    num_epochs = config['training']['epochs']
    for epoch in range(num_epochs):
        logging.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        # -- Training Phase --
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch in train_progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # Cast operations to mixed precision context manager
            with amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if loss is None:
                 logging.warning("Loss is None, skipping batch.")
                 continue

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})


        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                # Use autocast for validation too (though it doesn't affect gradients)
                # It ensures consistency and might slightly speed up validation inference
                with amp.autocast(enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                if loss is not None:
                    total_val_loss += loss.item()
                    val_progress_bar.set_postfix({'val_loss': loss.item()})


        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # -- Log metrics to CSV file --
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(training_log_file, "a") as f:
                f.write(f"{timestamp},{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
        except Exception as e:
            logging.error(f"Could not write to training log file: {e}")

        # -- Early Stopping Check --
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the model state dictionary (the weights)
            best_model_state = model.state_dict()
            logging.info("Validation loss improved. Saving best model state.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s). Patience: {patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Save the Best Model ---
    # Load the best state before saving if early stopping occurred and improved state was found
    if best_model_state:
        logging.info("Loading best model state before saving.")
        model.load_state_dict(best_model_state)
    elif epoch < num_epochs -1 : # Only warn if we stopped early but didn't save a best state
         logging.warning("Early stopping triggered, but no improvement detected over initial state. Saving the last state.")
    else:
        logging.info("Training finished. Saving the final model state.")


    try:
        model.save_pretrained(config['model']['classifier_path'])
        tokenizer.save_pretrained(config['model']['tokenizer_path'])
        logging.info(f"Model and tokenizer saved to {config['model']['classifier_path']}")
    except Exception as e:
        logging.error(f"Error saving model/tokenizer: {e}")


if __name__ == '__main__':
    from utils.helpers import load_config, create_directories # Corrected import
    config = load_config()
    if config:
        # Directories are likely created by main.py, but ensure they exist
        create_directories(config)
        main(config)

