import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW # Corrected import
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
from utils.helpers import load_data
import os
from datetime import datetime

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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main(config):
    """
    Main function to fine-tune the DistilBERT model with early stopping.
    """
    # --- Setup Training Log ---
    training_log_file = setup_training_log()

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load and Split Data ---
    df = load_data(config['data']['processed_data_path'])
    if df is None:
        return

    # Handle subsetting for faster testing runs
    if config['training']['training_subset_size'] != -1:
        df = df.sample(n=config['training']['training_subset_size'], random_state=config['preprocessing']['random_state'])
        logging.info(f"Using a subset of {len(df)} training examples.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state']
    )

    # --- Tokenization ---
    logging.info(f"Loading tokenizer: {config['model']['base_model']}")
    tokenizer = DistilBertTokenizer.from_pretrained(config['model']['base_model'])

    logging.info("Tokenizing training and testing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config['preprocessing']['max_length'])

    # --- Create Datasets and Dataloaders ---
    train_dataset = SuicidalTextDataset(train_encodings, train_labels)
    test_dataset = SuicidalTextDataset(test_encodings, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    # --- Model and Optimizer ---
    logging.info(f"Loading base model: {config['model']['base_model']}")
    model = DistilBertForSequenceClassification.from_pretrained(config['model']['base_model'], num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # --- Early Stopping Initialization ---
    patience = int(config['training'].get('early_stopping_patience', 3))
    min_delta = float(config['training'].get('min_delta', 0.01))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        logging.info(f"--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # -- Log metrics to CSV file --
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(training_log_file, "a") as f:
            f.write(f"{timestamp},{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # -- Early Stopping Check --
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save the best model state
            logging.info("Validation loss improved. Saving model state.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Save the Best Model ---
    if best_model_state:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state before saving.")
    else:
        logging.warning("No best model state saved; saving the last model state.")
        
    model.save_pretrained(config['model']['classifier_path'])
    tokenizer.save_pretrained(config['model']['tokenizer_path'])
    logging.info(f"Model and tokenizer saved to {config['model']['classifier_path']}")

if __name__ == '__main__':
    from utils.helpers import load_config, create_dirs
    config = load_config()
    if config:
        create_dirs(config)
        main(config)

