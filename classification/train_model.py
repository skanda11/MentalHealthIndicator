import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging
from tqdm import tqdm
from utils.helpers import load_config, create_directories
from datetime import datetime
import os

# (setup_training_log function remains the same)
def setup_training_log():
    """Sets up a CSV log file for training metrics."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,epoch,avg_train_loss,avg_val_loss\n")
    return log_file

def main(config):
    """
    Main function to fine-tune the DistilBERT model using pre-tokenized data.
    """
    training_log_file = setup_training_log()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Pre-Tokenized Data ---
    tokenized_data_path = config['data']['tokenized_data_path']
    try:
        logging.info(f"Loading pre-tokenized data from {tokenized_data_path}...")
        tokenized_data = torch.load(tokenized_data_path)
        dataset = TensorDataset(
            tokenized_data['input_ids'],
            tokenized_data['attention_mask'],
            tokenized_data['labels']
        )
    except FileNotFoundError:
        logging.error(f"Tokenized data file not found at {tokenized_data_path}. Please run the 'tokenize' step first.")
        return

    # --- Split Data ---
    test_size = int(len(dataset) * config['preprocessing']['test_size'])
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    logging.info(f"Data split into {len(train_dataset)} training and {len(test_dataset)} testing examples.")

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    # --- Model and Optimizer ---
    base_model_name = config['model']['base_model']
    logging.info(f"Loading base model: {base_model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- Training Loop with Early Stopping ---
    patience = int(config['training'].get('early_stopping_patience', 3))
    min_delta = float(config['training'].get('min_delta', 0.01))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config['training']['epochs']):
        logging.info(f"--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        
        # Training Phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1} Validation"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Log metrics and check for early stopping
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(training_log_file, "a") as f:
            f.write(f"{timestamp},{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            logging.info("Validation loss improved. Saving model state.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Save Model ---
    if best_model_state:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state before saving.")
    
    model.save_pretrained(config['model']['classifier_path'])
    # Also save the tokenizer used
    tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(config['model']['tokenizer_path'])
    logging.info(f"Model and tokenizer saved to {config['model']['classifier_path']}")


if __name__ == '__main__':
    config = load_config()
    if config:
        create_directories(config)
        main(config)

