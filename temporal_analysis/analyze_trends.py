import pandas as pd
import numpy as np
import logging
from utils.helpers import load_data, save_data
from utils.debug_logger import log_function_call

@log_function_call
def main(config):
    """
    Analyzes temporal trends in the prediction data.
    Since the dataset lacks timestamps, this function simulates them
    for demonstration purposes.
    """
    # Load predictions data
    df = load_data(config['data']['predictions_path'])
    if df is None:
        logging.error("Temporal analysis cannot continue without prediction data.")
        return

    # --- Simulate Timestamps ---
    logging.warning("No real timestamps in data. Simulating a date range for trend analysis.")
    start_date = config['temporal_analysis']['start_date']
    end_date = config['temporal_analysis']['end_date']
    num_records = len(df)
    
    # Generate random timestamps within the specified range
    date_range = pd.to_datetime(np.linspace(pd.to_datetime(start_date).value, pd.to_datetime(end_date).value, num_records))
    df['timestamp'] = date_range
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # --- Perform Trend Analysis ---
    logging.info("Performing temporal trend analysis...")
    df.set_index('timestamp', inplace=True)
    
    # Resample data by the frequency specified in the config (e.g., 'M' for month)
    resample_freq = config['temporal_analysis']['resample_freq']
    
    # Count total posts and suicide-risk posts per period
    monthly_counts = df['prediction'].resample(resample_freq).count().rename('total_posts')
    monthly_suicide_counts = df[df['prediction'] == 1]['prediction'].resample(resample_freq).count().rename('suicide_posts')
    
    # Combine into a single DataFrame
    trends_df = pd.concat([monthly_counts, monthly_suicide_counts], axis=1).fillna(0)
    
    # Calculate the proportion of suicide-risk posts
    trends_df['suicide_proportion'] = trends_df['suicide_posts'] / trends_df['total_posts']
    trends_df['suicide_proportion'].fillna(0, inplace=True) # Handle cases where total_posts is 0
    
    # --- Save Trend Analysis Results ---
    # Reset index to have 'timestamp' as a column for easier plotting
    trends_df.reset_index(inplace=True)
    save_data(trends_df, config['data']['trends_path'])
    logging.info(f"Temporal trend analysis results saved to {config['data']['trends_path']}")

if __name__ == '__main__':
    from utils.helpers import load_config
    config = load_config()
    main(config)


