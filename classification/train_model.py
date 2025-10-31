import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification
)
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import load_config

# --- Helper function (copied from your app.py) ---
def get_model_and_tokenizer_classes(model_name):
    if 'albert' in model_name.lower():
        return AlbertForSequenceClassification, AlbertTokenizer
    elif 'mobilebert' in model_name.lower():
         return MobileBertForSequenceClassification, MobileBertTokenizer
    else: # Default to DistilBERT
        return DistilBertForSequenceClassification, DistilBertTokenizer

# --- Main Training Function ---
def train_model():
    # 1. --- Load Config and Setup Paths ---
    print("Loading configuration...")
    config = load_config()
    if not config:
        print("Error: Could not load config.yaml.")
        return

    model_config = config['model']
    train_config = config['training'] # Assuming you have a 'training' section

    # Path for the BEST model (for prediction/dashboard)
    best_model_path = model_config['classifier_path']
    
    # Paths for RESUMABLE checkpoints
    checkpoint_dir = model_config['checkpoint']['checkpoint_dir']
    latest_checkpoint_path = os.path.join(checkpoint_dir, model_config['checkpoint']['latest_checkpoint_file'])
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. --- Setup Model, Tokenizer, and Optimizer ---
    print(f"Loading base model: {model_config['base_model']}")
    model_class, tokenizer_class = get_model_and_tokenizer_classes(model_config['base_model'])
    
    # Load the base model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_config['base_model'])
    model = model_class.from_pretrained(model_config['base_model'], num_labels=2) # Assuming 2 labels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup optimizer
    # --- !! REPLACE with your optimizer setup !! ---
    optimizer = optim.AdamW(model.parameters(), lr=train_config.get('learning_rate', 5e-5))
    
    # --- !! REPLACE with your data loading logic !! ---
    # (This is placeholder logic)
    print("Loading data... (REPLACE WITH YOUR DATALOADERS)")
    # train_dataset = ... 
    # val_dataset = ...
    # train_dataloader = DataLoader(train_dataset, batch_size=train_config.get('batch_size', 16))
    # val_dataloader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 16))
    
    
    # 3. --- Load Checkpoint if it Exists ---
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(latest_checkpoint_path):
        try:
            print(f"Loading checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start on the next epoch
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"Successfully resumed training from epoch {start_epoch}.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Starting from scratch. Error: {e}")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print("No checkpoint found. Starting training from scratch.")

    # 4. --- Training Loop ---
    num_epochs = train_config.get('num_epochs', 3)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1} / {num_epochs} ---")
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        # --- !! REPLACE with your training loop !! ---
        # for batch in tqdm(train_dataloader, desc="Training"):
        #     optimizer.zero_grad()
        #     inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        #     labels = batch['labels'].to(device)
        #     
        #     outputs = model(**inputs, labels=labels)
        #     loss = outputs.loss
        #     total_train_loss += loss.item()
        #     
        #     loss.backward()
        #     optimizer.step()
        # avg_train_loss = total_train_loss / len(train_dataloader)
        # print(f"Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            # --- !! REPLACE with your validation loop !! ---
            # for batch in tqdm(val_dataloader, desc="Validating"):
            #     inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            #     labels = batch['labels'].to(device)
            #     
            #     outputs = model(**inputs, labels=labels)
            #     loss = outputs.loss
            #     total_val_loss += loss.item()
        
        # avg_val_loss = total_val_loss / len(val_dataloader)
        # print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # This is placeholder. Replace with your actual validation loss.
        avg_val_loss = 0.5 # !! DUMMY VALUE !!
        
        # 5. --- Save Checkpoints ---
        
        # (A) Save BEST model for prediction
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
            
            # Use .save_pretrained() so app.py can load it
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            
        # (B) Save LATEST checkpoint for resumption
        try:
            print(f"Saving resumable checkpoint to {latest_checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, latest_checkpoint_path)
        except Exception as e:
            print(f"Error saving resumable checkpoint: {e}")

    print("Training finished.")

# if __name__ == "__main__":
#     train_model()