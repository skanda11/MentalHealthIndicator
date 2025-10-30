import streamlit as st
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification
)
import os
import sys
import random

# --- System and Configuration Setup ---
# Add project root to the Python path to allow importing utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from utils.helpers import load_config

# Load configuration at the start
config = load_config()
if not config:
    st.error("Fatal: Could not load configuration from config.yaml. Chatbot cannot start.", icon="🚨")
    st.stop()

# --- Helper function to get correct model classes ---
def get_model_and_tokenizer_classes(model_name):
    if 'albert' in model_name.lower():
        return AlbertForSequenceClassification, AlbertTokenizer
    elif 'mobilebert' in model_name.lower():
         return MobileBertForSequenceClassification, MobileBertTokenizer
    else: # Default to DistilBERT
        return DistilBertForSequenceClassification, DistilBertTokenizer

# --- Load Local Model ---
@st.cache_resource
def load_local_model(_config):
    """Load the fine-tuned local model and tokenizer."""
    try:
        model_name = _config['model']['base_model']
        tokenizer_path = _config['model']['tokenizer_path']
        model_path = _config['model']['classifier_path']

        model_class, tokenizer_class = get_model_and_tokenizer_classes(model_name)
        
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
             st.error(f"Model/Tokenizer not found at {model_path}. Please run the training pipeline first (`python main.py train`).")
             return None, None

        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(model_path)
        
        model.eval()  # Set model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading local model: {e}.")
        return None, None

def classify_text_local(text, _model, _tokenizer):
    """Classify text using the local fine-tuned model."""
    if not _model or not _tokenizer:
        return -1, 0.0 # Return -1 for error

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()
    return prediction, confidence

# --- Pre-defined Responses ---
# These are the questions the bot will ask based on the model's classification
RESPONSES = {
    "welcome": "Hi, I'm here to listen. How are you feeling today?",
    "neutral": [
        "Thanks for sharing. What else is on your mind?",
        "I understand. Tell me more.",
        "That's interesting. What else are you thinking about?",
        "Okay, I'm listening. What's next?"
    ],
    "risk_detected": [
        "It sounds like you're going through a very difficult time. I'm here to listen. Can you tell me more about what's happening?",
        "I'm hearing that you're in a lot of pain. It's brave of you to share that. What's been on your mind?",
        "That sounds incredibly tough. Please know that I'm listening. What's contributing to these feelings?"
    ],
    "helpline": (
        "It sounds like you are in significant distress. Please consider reaching out for immediate support. "
        "You can connect with the Vandrevala Foundation at **+91 9999666555**. They are available 24/7 to listen. "
        "Your well-being is important."
    )
}

# --- Streamlit App ---
st.set_page_config(page_title="Local Model Chat", page_icon="❤️‍🩹")
st.title("Local Model Assistant ")
st.caption("This chatbot uses your locally-trained classification model to guide the conversation.")

# Load the model
model, tokenizer = load_local_model(config)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add the first welcome message
    st.session_state.messages.append({"role": "assistant", "content": RESPONSES['welcome']})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How are you feeling?"):
    if not model or not tokenizer:
        st.error("Model is not loaded. Cannot process input.")
    else:
        # 1. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get classification from your local model
        with st.spinner("Analyzing..."):
            prediction, confidence = classify_text_local(prompt, model, tokenizer)

        # 3. Choose a response based on the classification
        if prediction == 1:
            # If risk is detected, pick a response from the 'risk_detected' list
            response_text = random.choice(RESPONSES['risk_detected'])
            
            # If confidence is very high, also add the helpline
            if confidence > 0.90:
                response_text += "\n\n" + RESPONSES['helpline']
        
        elif prediction == 0:
            # If no risk is detected, pick a response from the 'neutral' list
            response_text = random.choice(RESPONSES['neutral'])
        
        else:
            # If there was an error (prediction == -1)
            response_text = "I'm sorry, I had trouble processing that. Could you try rephrasing?"

        # 4. Add assistant's response to history and display it
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)