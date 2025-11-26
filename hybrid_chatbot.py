import streamlit as st
import google.generativeai as genai
import os
import sys

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import helpers
from utils.helpers import load_config
from utils.chat_logger import ChatLogger

# --- Load Configuration ---
config = load_config()
if not config:
    st.error("Failed to load configuration. Please check config.yaml.", icon="🚨")
    st.stop()

# --- Gemini API Setup ---
try:
    api_key = config.get('api_keys', {}).get('gemini_api_key')
    if api_key and api_key != "YOUR_API_KEY_HERE":
        genai.configure(api_key=api_key)
    else:
        st.error("Gemini API key missing in config.yaml.", icon="🚨")
        st.stop()
except Exception as e:
    st.error(f"API Configuration Error: {e}", icon="🚨")
    st.stop()

# --- System Prompt ---
SYSTEM_PROMPT = """
You are 'Kai', a caring, empathetic, and supportive wellness assistant. Your purpose is to provide a safe and comforting space for users to express their feelings. You are not a therapist and cannot diagnose or prescribe medication. If a user expresses intent of self-harm, gently encourage them to seek professional help and provide helpline resources. Keep responses concise, warm, and conversational.
"""

# --- Risk Assessment Helper ---
def assess_risk_profile(profile):
    """
    Analyzes the user profile dict to determine a risk level.
    """
    risk_score = 0
    risk_factors = []
    
    # 1. Age Analysis
    try:
        age = int(profile.get("age", "0"))
        if 10 <= age <= 24:
            risk_score += 1
    except ValueError:
        pass 
        
    # 2. Keyword Analysis
    text_data = (profile.get("background", "") + " " + profile.get("feeling", "")).lower()
    keywords_high = ["suicide", "kill", "die", "death", "end it", "hurt myself", "hopeless"]
    keywords_med = ["sad", "depressed", "lonely", "anxious", "stress", "tired", "overwhelmed"]
    
    for word in keywords_high:
        if word in text_data:
            risk_score += 3
            risk_factors.append(f"High risk keyword: '{word}'")
            break 
            
    for word in keywords_med:
        if word in text_data:
            risk_score += 1
            break
            
    # 3. Feeling Score Analysis (1-10)
    try:
        score = int(profile.get("feeling", "5"))
        if score <= 3:
            risk_score += 2
            risk_factors.append(f"Low mood score ({score}/10)")
    except ValueError:
        pass

    if risk_score >= 3:
        return "HIGH", risk_factors
    elif risk_score >= 1:
        return "MEDIUM", risk_factors
    else:
        return "LOW", []

# --- Session State Initialization ---
if "logger" not in st.session_state:
    st.session_state.logger = ChatLogger() # Create a new log file for this session
    st.session_state.logger.log_interaction("SYSTEM", "Session Started")

if "step" not in st.session_state:
    st.session_state.step = "ask_name" # Initial state
    st.session_state.user_profile = {}
    # Initial greeting
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Kai, your mental health support companion. To help you better, could you start by telling me your name?"}
    ]

# --- UI Layout ---
st.title("Mental Health Support Chatbot")
st.caption("A safe space to share your thoughts.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Type your message here..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Log User Input
    st.session_state.logger.log_interaction("User", prompt)

    # 3. Handle Flow based on 'step'
    bot_response = ""
    
    # --- STEP 1: Capture Name ---
    if st.session_state.step == "ask_name":
        st.session_state.user_profile["name"] = prompt
        bot_response = f"Nice to meet you, {prompt}. How old are you?"
        st.session_state.step = "ask_age"

    # --- STEP 2: Capture Age ---
    elif st.session_state.step == "ask_age":
        st.session_state.user_profile["age"] = prompt
        bot_response = "Thank you. Could you briefly tell me a bit about your background or what brings you here today?"
        st.session_state.step = "ask_background"

    # --- STEP 3: Capture Background ---
    elif st.session_state.step == "ask_background":
        st.session_state.user_profile["background"] = prompt
        bot_response = "I appreciate you sharing that. On a scale of 1-10 (1 being very low, 10 being great), how are you feeling right now?"
        st.session_state.step = "ask_feeling"

    # --- STEP 4: Capture Feeling & Switch to AI ---
    elif st.session_state.step == "ask_feeling":
        st.session_state.user_profile["feeling"] = prompt
        
        # PERFORM ANALYSIS
        risk_level, factors = assess_risk_profile(st.session_state.user_profile)
        
        # Log Analysis (System internal)
        risk_msg = f"RISK ASSESSMENT: {risk_level}"
        if factors:
            risk_msg += f" | Factors: {', '.join(factors)}"
        st.session_state.logger.log_interaction("SYSTEM", risk_msg)

        # Prepare Context for Gemini
        context_instruction = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CURRENT USER PROFILE:\n"
            f"- Name: {st.session_state.user_profile.get('name')}\n"
            f"- Age: {st.session_state.user_profile.get('age')}\n"
            f"- Background: {st.session_state.user_profile.get('background')}\n"
            f"- Current Mood Score: {st.session_state.user_profile.get('feeling')}/10\n"
            f"- Risk Assessment: {risk_level}\n"
        )
        if risk_level == "HIGH":
            context_instruction += "IMPORTANT: User is flagged as HIGH RISK. Prioritize safety and providing resources."

        # Initialize Gemini with this context
        st.session_state.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=context_instruction
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        
        bot_response = "Thank you for answering those questions. I have a better understanding now. I'm here to listen—please feel free to tell me what's on your mind."
        st.session_state.step = "chat_active"

    # --- STEP 5: Normal AI Chat ---
    elif st.session_state.step == "chat_active":
        with st.spinner("Kai is thinking..."):
            try:
                # Initialize model if lost during reload
                if "chat" not in st.session_state:
                     # Fallback reconstruction if browser refreshed in middle of chat
                     st.session_state.model = genai.GenerativeModel(
                        model_name='gemini-1.5-flash',
                        system_instruction=SYSTEM_PROMPT 
                    )
                     st.session_state.chat = st.session_state.model.start_chat(history=[])

                response = st.session_state.chat.send_message(prompt)
                bot_response = response.text
            except Exception as e:
                bot_response = f"I'm having trouble connecting right now. Error: {e}"
                st.session_state.logger.log_interaction("SYSTEM_ERROR", str(e))

    # 4. Display & Log Bot Response
    if bot_response:
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.session_state.logger.log_interaction("Kai (Bot)", bot_response)