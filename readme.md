# Mental Health Indicator Analysis Pipeline

## 1. Overview

This project implements an end-to-end pipeline to analyze text data for indicators of mental health risks, specifically focusing on suicide ideation. It uses a fine-tuned DistilBERT model to classify text, performs temporal trend analysis on the predictions, and presents the findings in an interactive Streamlit dashboard.

The primary goal is to provide a framework for identifying patterns and trends in textual data that could be indicative of mental health crises, offering a valuable tool for researchers, mental health professionals, and platform moderators.

## 2. Project Structure

The codebase is organized into modular components, each handling a specific part of the pipeline:

```bash
MentalHealthIndicator/
├── config.yaml             # Project configuration (paths, model params)
├── main.py                 # Main script to run the entire pipeline
├── README.md               # This readme file
├── requirements.txt        # Python dependencies
│
├── utils/                  # Utility functions
│   └── helpers.py          # Config loading, directory creation, data I/O
│
├── preprocessing/          # Data preprocessing module
│   └── preprocess.py       # Cleans text and labels data
│
├── classification/         # NLP model classification module
│   ├── train_model.py      # Fine-tunes the DistilBERT model
│   └── predict.py          # Uses the trained model for predictions
│
├── temporal_analysis/      # Temporal analysis module
│   └── analyze_trends.py   # Analyzes prediction trends over time
│
├── dashboard/              # Streamlit dashboard module
│   └── app.py              # Interactive dashboard for results
│
├── data/                   # Directory for all data
│   ├── raw/                # Raw input data (e.g., Suicide_Detection.csv)
│   ├── processed/          # Processed data after cleaning
│   └── predictions/        # Model predictions and trend analysis results
│
└── models/                 # Directory for trained models
    └── classifier/         # Stores the tokenizer and trained model weights
```

## 3. Setup and Installation

**Clone the Repository (if applicable) or ensure all project files are in the MentalHealthIndicator/ directory.** 

```bash
git clone https://github.com/skanda11/MentalHealthIndicator.git
```

**Install Dependencies: Install the required packages from the requirements.txt file.**

```bash
pip install -r requirements.txt
```
***Download Data :***
Download the dataset and place it in a directory. Update the `data_path` in `config/config.yaml` to point to dataset location (e.g., Suicide_Detection.csv is inside the data/raw/ directory.)



## 4. How to Run the Pipeline

The main.py script is the central entry point for running the pipeline. Navigate to the project's root directory (MentalHealthIndicator/) in your terminal to run the following commands.

**Run the Entire Pipeline**

To run all steps from preprocessing to temporal analysis sequentially, use the all argument:
```bash
python main.py all
```

**Run Specific Steps**

You can also run individual steps as needed. This is useful for re-running a specific part of the process without starting from scratch.

***Preprocess the data:***
```bash
python main.py preprocess
```

***Train the model:***
```bash
python main.py train
```

***Generate predictions:***
```bash
python main.py predict
```

***Analyze temporal trends:***
```bash
python main.py analyze
```

***You can also combine steps:***

*Example: Preprocess data and then train the model*
```bash
python main.py preprocess train
```

## 5. How to View the Dashboard

After running the pipeline (at least up to the analyze step), you can launch the interactive dashboard to visualize the results.

Run the following command from the project's root directory:
```bash
streamlit run dashboard/app.py
```

This will start a local web server and provide a URL (usually http://localhost:8501) that you can open in your web browser to view and interact with the dashboard.