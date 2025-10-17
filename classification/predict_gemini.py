import pandas as pd
import logging
import time
import json
import google.generativeai as genai
from tqdm import tqdm
from utils.helpers import load_data, save_data
from utils.debug_logger import log_function_call

def get_gemini_classification(model, text_batch):
    """
    Sends a batch of texts to the Gemini API for classification,
    with an exponential backoff retry mechanism to handle rate limits.
    """
    # Create a single prompt with all texts in the batch
    prompt = "Analyze each of the following texts for suicidal ideation. Respond with a valid JSON array where each object has a 'text' key with the original text, and a 'prediction' key which is an integer (1 for 'Suicide Risk Detected', 0 for 'No Suicide Risk Detected').\n\n---\n"
    for text in text_batch:
        prompt += f"- {json.dumps(text)}\n"
    prompt += "---"

    retries = 5
    delay = 2  # Start with a 2-second delay

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            # Clean the response to extract the JSON part
            json_text = response.text.strip().replace('```json', '').replace('```', '')
            predictions = json.loads(json_text)
            
            # Create a mapping from text to prediction for easy lookup
            prediction_map = {item['text']: item['prediction'] for item in predictions}
            
            # Return predictions in the same order as the input batch
            return [prediction_map.get(text, -1) for text in text_batch] # Default to -1 for errors

        except Exception as e:
            if "resource_exhausted" in str(e).lower() or "quota" in str(e).lower():
                logging.warning(f"Quota exhausted. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Double the delay for the next attempt
            else:
                logging.error(f"An unexpected error occurred with the Gemini API: {e}")
                return [-1] * len(text_batch) # Return error for the whole batch on non-quota errors

    logging.error("Failed to get response after multiple retries. Aborting this batch.")
    return [-1] * len(text_batch)

@log_function_call
def main(config):
    """
    Generates predictions using the Gemini API.
    """
    # --- Configure Gemini API ---
    try:
        api_key = config['api_keys']['gemini_api_key']
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            logging.error("Gemini API key not found in config.yaml. Please add it to run this script.")
            return
        genai.configure(api_key=api_key)
        # Use the more powerful model for higher accuracy
        model = genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        return

    # --- Load Data ---
    df = load_data(config['data']['processed_data_path'])
    if df is None:
        return

    texts = df['text'].astype(str).tolist()
    all_predictions = []
    
    # Process texts in batches to respect API rate limits
    batch_size = 20  # Number of texts to send in one API call
    
    logging.info(f"Starting prediction with Gemini API on {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting with Gemini"):
        batch_texts = texts[i:i + batch_size]
        predictions = get_gemini_classification(model, batch_texts)
        all_predictions.extend(predictions)
        
        # The static sleep is removed as the backoff logic now handles delays dynamically
        # time.sleep(1) 

    # --- Save Predictions ---
    df['prediction_gemini'] = all_predictions
    save_path = config['data']['gemini_predictions_path']
    save_data(df[['text', 'label', 'prediction_gemini']], save_path)
    logging.info(f"Gemini predictions saved to {save_path}")

if __name__ == '__main__':
    from utils.helpers import load_config
    config = load_config()
    if config:
        main(config)

