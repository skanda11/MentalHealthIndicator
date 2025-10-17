import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import sys
import google.generativeai as genai
import json

# --- System and Configuration Setup ---
# Add project root to the Python path to allow importing utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import load_config

# Load configuration at the start
config = load_config()
if not config:
    st.error("Fatal: Could not load configuration from config.yaml. Dashboard cannot start.", icon="🚨")
    st.stop()

# --- Gemini API Configuration ---
api_key = None
try:
    api_key = config.get('api_keys', {}).get('gemini_api_key')
    if api_key and api_key != "YOUR_API_KEY_HERE":
        genai.configure(api_key=api_key)
    else:
        api_key = None # Ensure api_key is None if not set properly
except Exception as e:
    st.warning(f"Could not configure Gemini API. The 'Second Opinion' feature will be unavailable. Error: {e}")

st.set_page_config(page_title="Mental Health Indicator Dashboard", layout="wide")

# --- Function Definitions ---
@st.cache_resource
def get_config_cached():
    """Return the already loaded config."""
    return config

@st.cache_resource
def load_model_and_tokenizer(_config):
    """Load the fine-tuned local model and tokenizer."""
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(_config['model']['tokenizer_path'])
        model = DistilBertForSequenceClassification.from_pretrained(_config['model']['classifier_path'])
        model.eval()  # Set model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading local model: {e}. Please ensure the model is trained and paths in config.yaml are correct.")
        return None, None

def classify_text_local(text, _model, _tokenizer):
    """Classify text using the local fine-tuned model."""
    if not _model or not _tokenizer:
        return "Model not loaded.", 0.0

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()
    label = "Suicide Risk Detected" if prediction == 1 else "No Suicide Risk Detected"
    return label, confidence

def get_gemini_response(text):
    """Get a classification and explanation from the Gemini API."""
    if not api_key:
         return {"error": "API key for Gemini is not configured in config.yaml."}

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analyze the following text for signs of suicidal ideation. Your response must be a valid JSON object.
        Do not include any text or markdown formatting before or after the JSON object.
        The JSON object should have two keys:
        1. "classification": a string, which must be either "Suicide Risk Detected" or "No Suicide Risk Detected".
        2. "reason": a string, providing a brief, one-sentence explanation for your classification.

        Text to analyze:
        ---
        {text}
        ---
        """
        
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
        
    except Exception as e:
        return {"error": f"An error occurred with the Gemini API: {e}"}

@st.cache_data
def load_analysis_data(trends_path, predictions_path):
    """Load the trend analysis and predictions data."""
    try:
        trends_df = pd.read_csv(trends_path)
        trends_df['timestamp'] = pd.to_datetime(trends_df['timestamp'])
        predictions_df = pd.read_csv(predictions_path)
        return trends_df, predictions_df
    except FileNotFoundError as e:
        st.warning(f"Data file not found: {e}. Please run the full pipeline first.")
        return None, None

# --- Main Application Logic ---
local_model, local_tokenizer = load_model_and_tokenizer(config)
trends_df, predictions_df = load_analysis_data(config['data']['trends_path'], config['data']['predictions_path'])

# --- UI Layout ---
st.title("Mental Health Indicator Analysis Dashboard")
st.markdown("This dashboard provides insights from a model trained to detect potential suicide risk in text posts. It also allows for real-time text classification using the local model and Google's Gemini Pro for a second opinion.")

tab1, tab2, tab3 = st.tabs(["📊 Trend Analysis", "🔍 Live Classifier", "📚 Data Explorer"])

# Tab 1: Trend Analysis
with tab1:
    st.header("Temporal Trends of Suicide Risk Posts")
    if trends_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_prop = px.line(
                trends_df, x='timestamp', y='suicide_proportion',
                title='Proportion of Suicide-Risk Posts Over Time',
                labels={'timestamp': 'Week', 'suicide_proportion': 'Proportion'},
                markers=True
            )
            fig_prop.update_layout(yaxis_tickformat=".2%")
            st.plotly_chart(fig_prop, use_container_width=True)
        with col2:
            fig_vol = px.bar(
                trends_df, x='timestamp', y=['total_posts', 'suicide_posts'],
                title='Volume of Posts Over Time',
                labels={'timestamp': 'Week', 'value': 'Number of Posts'},
                barmode='group'
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Trend data not available. Please run the `temporal_analysis` step.")

# Tab 2: Live Classifier
with tab2:
    st.header("Real-Time Text Classification")
    user_text = st.text_area("Enter text here for analysis:", height=150, placeholder="Type or paste text...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze with Local Model", use_container_width=True):
            if user_text:
                with st.spinner("Analyzing with local model..."):
                    label, confidence = classify_text_local(user_text, local_model, local_tokenizer)
                    st.session_state.local_result = (label, confidence)
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        if st.button("Get Second Opinion with Gemini", use_container_width=True, disabled=(not api_key)):
            if not api_key:
                st.error("Gemini API key not found. Please set it in your config.yaml file.")
            elif user_text:
                with st.spinner("Contacting Gemini for a second opinion..."):
                    gemini_result = get_gemini_response(user_text)
                    st.session_state.gemini_result = gemini_result
            else:
                st.warning("Please enter some text to analyze.")

    st.markdown("---")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Your Fine-Tuned Model's Result")
        if 'local_result' in st.session_state:
            label, confidence = st.session_state.local_result
            if "Risk Detected" in label:
                st.error(f"**Result:** {label}")
            else:
                st.success(f"**Result:** {label}")
            st.progress(confidence, text=f"Confidence: {confidence:.2%}")
    
    with res_col2:
        st.subheader("Gemini's Second Opinion")
        if 'gemini_result' in st.session_state:
            result = st.session_state.gemini_result
            if "error" in result:
                st.error(f"**Error:** {result['error']}")
            else:
                label = result.get('classification', 'N/A')
                reason = result.get('reason', 'No explanation provided.')
                if "Risk Detected" in label:
                    st.error(f"**Result:** {label}")
                else:
                    st.success(f"**Result:** {label}")
                st.info(f"**Reasoning:** {reason}")

# Tab 3: Data Explorer
with tab3:
    st.header("Explore Processed Data and Predictions")
    if predictions_df is not None:
        st.dataframe(predictions_df)
    else:
        st.info("Prediction data not available. Please run the `predict` step.")

