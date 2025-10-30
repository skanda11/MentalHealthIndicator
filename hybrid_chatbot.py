import streamlit as st
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification
)
import os
import sys
import google.generativeai as genai
import json
import logging

# --- System and Configuration Setup ---
# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from utils.helpers import load_config
from utils.debug_logger import setup_logger

# Setup logger
setup_logger()

# Load configuration
config = load_config()
if not config:
    st.error("Fatal: Could not load configuration from config.yaml. Chatbot cannot start.", icon="🚨")
    st.stop()

# --- Gemini API Configuration ---
try:
    api_key = config.get('api_keys', {}).get('gemini_api_key')
    if api_key and api_key != "YOUR_API_KEY_HERE":
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    else:
        st.error("Your Gemini API key is not configured in config.yaml. Please add it to run the chatbot.", icon="🚨")
        st.stop()
except Exception as e:
    st.error(f"There was an error configuring the Gemini API: {e}", icon="🚨")
    st.stop()

# --- Load Local Classification Model ---
@st.cache_resource
def get_model_and_tokenizer_classes(model_name):
    """Gets the correct model classes from the model name in the config."""
    if 'albert' in model_name.lower():
        return AlbertForSequenceClassification, AlbertTokenizer
    elif 'mobilebert' in model_name.lower():
         return MobileBertForSequenceClassification, MobileBertTokenizer
    else: # Default to DistilBERT
        return DistilBertForSequenceClassification, DistilBertTokenizer

@st.cache_resource
def load_local_model(_config):
    """Load the fine-tuned local model and tokenizer."""
    try:
        model_name = _config['model']['base_model']
        tokenizer_path = _config['model']['tokenizer_path']
        model_path = _config['model']['classifier_path']

        model_class, tokenizer_class = get_model_and_tokenizer_classes(model_name)
        
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
             logging.error(f"Model/Tokenizer not found at {model_path}. Please run the training pipeline first (`python main.py train`).")
             return None, None

        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(model_path)
        
        model.eval()
        logging.info("Local classification model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading local model: {e}.")
        return None, None

def classify_text_local(text, _model, _tokenizer):
    """Classify text using the local fine-tuned model. Returns 'Risk' or 'No Risk'."""
    if not _model or not _tokenizer:
        return "Error", 0.0

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()
    label = "Risk Detected" if prediction == 1 else "No Risk Detected"
    return label, confidence

# --- Streamlit App ---

st.set_page_config(page_title="Kai - Hybrid AI Assistant", page_icon="🧠")
st.title("Kai (Hybrid Mode) 🧠")
st.caption("Using a local classifier to inform a generative AI response.")

# Load the local model
local_model, local_tokenizer = load_local_model(config)

# Initialize chat history
if "chat" not in st.session_state:
    # We provide the system prompt to Gemini, telling it how to behave
    # and how to interpret the classifier's output.
    system_prompt = """
    You are 'Kai', an empathetic and supportive wellness assistant. Your goal is to hold a natural, 
    caring conversation. 
    
    You will receive user input in two parts:
    1.  [Internal Classification]: A tag that is either [No Risk] or [Risk Detected]. This is from a 
        separate AI model and is for YOUR information only. DO NOT repeat this tag to the user.
    2.  [User Message]: The raw text the user wrote.

    Your job is to respond ONLY to the [User Message] in a conversational way.
    
    - If the classification is [No Risk], just continue the conversation naturally and supportively.
    - If the classification is [Risk Detected], your tone should become more gentle and concerned. 
      Ask open-ended questions to understand their feelings better (e.g., "That sounds really tough, 
      can you tell me more about what's on your mind?").
    - If the user explicitly mentions suicide, self-harm, or being in a crisis, you MUST provide 
      this helpline resource: "It sounds like you are going through a very difficult time. 
      Please consider reaching out for immediate support. You can connect with the Vandrevala 
      Foundation at +91 9999666555. They are available 24/7 to listen."
    - Do NOT give medical advice. Be a listener.
    """
    st.session_state.chat = gemini_model.start_chat(history=[])
    st.session_state.messages = []
    
    # Add the first welcome message
    welcome_msg = "Hi, I'm Kai. I'm here to listen. How are you feeling today?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How are you feeling?"):
    if not local_model or not local_tokenizer:
        st.error("Local model is not loaded. Cannot process input.")
    else:
        # 1. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get classification from your local model (silently)
        classification, confidence = classify_text_local(prompt, local_model, local_tokenizer)
        logging.info(f"Local Model Classification: {classification} (Confidence: {confidence:.2f})")

        # 3. Create a new, combined prompt for Gemini
        hybrid_prompt = f"[Internal Classification: {classification}] \n[User Message]: {prompt}"
        
        # 4. Get generative response from Gemini
        with st.spinner("Kai is thinking..."):
            try:
                response = st.session_state.chat.send_message(hybrid_prompt)
                response_text = response.text
                
            except Exception as e:
                logging.error(f"Error calling Gemini API: {e}")
                response_text = "I'm sorry, I'm having a little trouble connecting right now. Please try again in a moment."

        # 5. Add assistant's response to history and display it
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)