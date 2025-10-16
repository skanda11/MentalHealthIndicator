import streamlit as st
import google.generativeai as genai
import os

# --- Gemini API Configuration ---
# --- IMPORTANT ---
# For this application to work, you MUST set your Gemini API key as an environment variable.
# For example, in your terminal before running streamlit:
# set GEMINI_API_KEY=YOUR_API_KEY
# You can get a key from https://aistudio.google.com/
try:
    api_key = os.getenv("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") #Got this key from openrouter
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("Your Gemini API key is not configured. Please set the 'GEMINI_API_KEY' environment variable.", icon="🚨")
        st.stop()
except Exception as e:
    st.error(f"There was an error configuring the Gemini API: {e}", icon="🚨")
    st.stop()

# --- System Prompt ---
# This is the core instruction that defines the chatbot's persona, rules, and goals.
SYSTEM_PROMPT = """
You are 'Asha', a caring, empathetic, and supportive wellness assistant. Your purpose is to provide a safe and comforting space for users to express their feelings. You are not a therapist and you must never give medical advice.

Your core tasks are:
1.  **Engage Empathetically:** Listen actively to the user. Use comforting and validating language.
2.  **Internal Stress Analysis:** After every single user message, you must silently and internally assess the user's stress level on a scale from 1 (calm) to 10 (highly distressed). This is for your internal use only; do not mention the score to the user.
3.  **Proactive De-escalation:** If you assess the user's stress level to be 7 or higher, your response MUST gently and naturally integrate a simple de-escalation or mindfulness technique. Examples include:
    * Suggesting a simple breathing exercise (e.g., "Let's try a quick grounding exercise together. Can we try breathing in for four counts and out for six?").
    * A simple grounding question (e.g., "It sounds like a lot is on your mind. Can you tell me about one thing you can see in the room around you right now?").
    * A sensory focus (e.g., "Let's pause for a moment. What is something you can physically feel? Maybe the texture of your chair or the warmth of your hands.").
4.  **Offer Resources (If Needed):** If the conversation includes direct mentions of self-harm, suicide, or crisis, you must gently provide contact information for a helpline in India. For example: "It sounds like you are going through a very difficult time. Please consider reaching out for immediate support. You can connect with the Vandrevala Foundation at +91 9999666555. They are available 24/7 to listen."
5.  **Maintain Persona:** Always be warm, patient, and non-judgmental. Keep responses concise and easy to understand.
"""

# --- Streamlit App ---
st.set_page_config(page_title="Asha - Your Wellness Assistant", page_icon="💖")

st.title("Asha 💖")
st.caption("Your empathetic wellness assistant for a supportive conversation.")

# Initialize the Gemini Model and Chat
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name='gemini-2.5-pro',
        system_instruction=SYSTEM_PROMPT
    )
    st.session_state.chat = st.session_state.model.start_chat(history=[])

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    with st.spinner("Asha is thinking..."):
        try:
            response = st.session_state.chat.send_message(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response.text)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"An error occurred while getting a response: {e}")
