# Mental Health Indicator Analysis Pipeline

## 1. Overview

This project implements an end-to-end pipeline to analyze text data for indicators of mental health risks, specifically focusing on suicide ideation. It uses a fine-tuned DistilBERT model to classify text, performs temporal trend analysis on the predictions, and presents the findings in an interactive Streamlit dashboard.

The primary goal is to provide a framework for identifying patterns and trends in textual data that could be indicative of mental health crises, offering a valuable tool for researchers, mental health professionals, and platform moderators.

## 2. Project Structure

The codebase is organized into modular components, each handling a specific part of the pipeline:

```bash
MentalHealthIndicator/
├── models/
│   └── 1_classifier/         # Stores the trained model and tokenizer
├── data/
│   ├── raw/                  # Holds raw input data (e.g., Suicide_Detection.csv)
│   ├── processed/            # Stores cleaned data and tokenized tensors
│   └── predictions/          # Stores model predictions and analysis results
├── preprocessing/
│   └── preprocess.py         # Cleans and labels raw text data
├── tokenization/
│   └── tokenize_data.py      # Converts processed text into tensors for the model
├── classification/
│   ├── train_model.py        # Fine-tunes the transformer model
│   ├── predict.py            # Generates predictions using the trained model
│   └── evaluate_model.py     # Calculates accuracy, precision, recall, F1
├── temporal_analysis/
│   └── analyze_trends.py     # Analyzes prediction trends over time
├── dashboard/
│   └── dashboard.py          # Streamlit app for data visualization and live testing
├── chatbot.py                # Streamlit app: 100% Generative AI (Gemini) chatbot
├── hybrid_chatbot.py         # Streamlit app: Local model + Generative AI chatbot
├── utils/
│   ├── helpers.py            # Config loading, data I/O, etc.
│   └── debug_logger.py       # Configures logging for the project
├── logs/
│   ├── run_YYYY-MM-DD.log    # Main log file
│   └── evaluation_results.txt # Output from the 'evaluate' step
│
├── config.yaml               # Master configuration file
├── main.py                   # Main script to run the entire pipeline
├── requirements.txt          # Python dependencies
└── GPU Setup Guide.md        # Instructions for setting up a GPU
```

## 3. Setup and Installation

**Clone the Repository (if applicable) or ensure all project files are in the MentalHealthIndicator/ directory.** 

```bash
git clone https://github.com/skanda11/MentalHealthIndicator.git
cd MentalHealthIndicator
```

**Install Dependencies: Install the required packages from the requirements.txt file.**

```bash
pip install -r requirements.txt
```
***Download Data :***
Download the dataset and place it in a directory. Update the `data_path` in `config/config.yaml` to point to dataset location (e.g., Suicide_Detection.csv is inside the data/raw/ directory.)

****Add API Keys:  Open the config.yaml file and add your API key for the generative model you wish to use (Google Gemini or Anthropic Claude).****
```bash
api_keys: 
  gemini_api_key: "YOUR_GEMINI_API_KEY_HERE"
  anthropic_api_key: "YOUR_ANTHROPIC_API_KEY_HERE" # This was not used in my project. 

  # Set the model for the hybrid chatbot
  # Options: "gemini-2.5-flash", "gemini-2.5-pro", "claude-3-haiku-20240307"
  generative_model_name: "gemini-2.5-flash" 
```
#### (Optional) GPU Setup Training is significantly faster on a GPU. If you have an NVIDIA GPU, follow the instructions in "GPU Setup Guide.md" to install the correct version of PyTorch with CUDA support.

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

***Tokenize: Converts cleaned text into tensors and saves them to*** ```data/processed/tokenized_data.pt```

```bash
python main.py tokenize
```


***Train the model: Fine-tunes the model (specified in ```config.yaml```) on the tokenized data*** 
```bash
python main.py train
```

***Generate predictions: Uses the trained model to generate predictions on the test set.***
```bash
python main.py predict
```

***GEvaluate: Compares predictions to the true labels and generates a report. Results are saved to*** ```logs/evaluation_results.txt``` and ```logs/confusion_matrix.png.```
```bash
python main.py evaluate
```

***Analyze temporal trends: Simulates timestamps and analyzes trends in the predictions.***
```bash
python main.py analyze
```

***You can also combine steps:***

*Example: Preprocess data and then train the model*
```bash
python main.py preprocess train
```

## 5. Running the Applications

### This project includes two separate Streamlit applications.

After running the pipeline (at least up to the analyze step), you can launch the interactive dashboard to visualize the results.

# A. Data & Model Dashboard

This dashboard visualizes the results of your pipeline, including temporal trends and a "second opinion" tool to compare your local model against the Gemini API.

**Prerequisite:** You must run the **all** or **analyze** step at least once.

To launch:
```bash
python main.py dashboard
```

# B. Hybrid "Kai" Chatbot
This is an advanced chatbot that uses both your locally-trained classifier and a powerful generative LLM (like Gemini or Claude). Your local model first "listens" to the user's input to detect risk, and this information is then fed to the generative model to provide a safe, natural, and empathetic response.

**Prerequisite:** You must run the **train** step at least once and have a valid API key in **config.yaml**.

To launch:
```bash
python main.py hybrid_chat
```


This will start a local web server and provide a URL (usually http://localhost:8501) that you can open in your web browser to view and interact with the dashboard.
<!--
## 6. Future Enhancements

The following features are planned for future development to enhance the project's capabilities and impact:

#### Hybrid AI Analysis with Gemini:

Integrate the Google Gemini API into the Streamlit dashboard.

This will add a **"Get a Second Opinion with Gemini"** button, allowing users to get a real-time, nuanced analysis from a state-of-the-art Large Language Model, complementing the prediction from the locally trained DistilBERT model.

#### Interactive Wellness Chatbot:

Develop a separate Streamlit application (chatbot.py) that functions as an empathetic wellness assistant.

The chatbot will use the Gemini API with a specialized system prompt to engage in supportive conversations, internally assess the user's stress level, and proactively suggest de-escalation or mindfulness techniques if it detects high levels of distress.

It will also be programmed to provide contact information for mental health helplines in India if a user expresses direct mentions of crisis.
>