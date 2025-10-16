import pandas as pd
import re
from sklearn.model_selection import train_test_split
import logging
from utils.helpers import load_data, save_data
from utils.debug_logger import log_function_call

def clean_text(text):
    """
    Cleans the input text by removing URLs, special characters, and extra whitespace.
    
    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def label_data(df):
    """
    Creates a numerical label from the 'class' column.
    'suicide' -> 1
    'non-suicide' -> 0
    
    Args:
        df (pandas.DataFrame): DataFrame with a 'class' column.
        
    Returns:
        pandas.DataFrame: DataFrame with a new 'label' column.
    """
    df['label'] = df['class'].apply(lambda x: 1 if x == 'suicide' else 0)
    return df

@log_function_call
def main(config):
    """
    Main function for the preprocessing step. Loads raw data, cleans it,
    labels it, and saves the processed data.
    """
    # Load raw data
    raw_df = load_data(config['data']['raw_data_path'])
    if raw_df is None:
        logging.error("Preprocessing cannot continue without raw data.")
        return

    # Clean the text
    logging.info("Cleaning text data...")
    raw_df['cleaned_text'] = raw_df['text'].apply(clean_text)
    
    # Create labels
    logging.info("Labeling data...")
    processed_df = label_data(raw_df)
    
    # Select relevant columns
    final_df = processed_df[['cleaned_text', 'label']].copy()
    final_df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    
    # Save processed data
    save_data(final_df, config['data']['processed_data_path'])
    logging.info(f"Processed data saved to {config['data']['processed_data_path']}")

if __name__ == '__main__':
    # This part allows running the script directly for testing
    from utils.helpers import load_config
    config = load_config()
    main(config)


