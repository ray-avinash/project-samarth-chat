# File: frontend/app.py

import streamlit as st
import requests
import json

# --- Backend Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000/chat"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Project Samarth", page_icon="🌾")
st.title("🌾 Project Samarth Q&A")
st.caption("I can answer questions about crop production and rainfall in India.")

# --- Chat History Management ---
if "messages" not in st.session_state:
    # st.session_state.messages will store list of dicts: {"role": "...", "content": "..."}
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Logic ---
if prompt := st.chat_input("What would you like to know?"):
    
    # Store the history *before* adding the new prompt
    # This is what we'll send to the backend
    history_for_backend = st.session_state.messages
    
    # 1. Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare for and send request to backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... ⏳")
        
        try:
            # Create the JSON payload to send to the API
            payload = {
                "query": prompt,
                "chat_history": history_for_backend 
            }
            
            # Make the POST request to the FastAPI backend
            response = requests.post(FASTAPI_URL, json=payload, timeout=120) # 120-second timeout for complex agents
            
            # Check for a successful response
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                answer = data.get("answer", "No answer found in response.")
                
                # Display the answer
                message_placeholder.markdown(answer)
                # 4. Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Handle HTTP errors
                message_placeholder.error(f"Error from backend: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Connection Error: Could not connect to the backend. Is it running?")
        except requests.exceptions.Timeout:
            message_placeholder.error("Error: The request timed out. The agent might be taking too long.")
        except Exception as e:
            message_placeholder.error(f"An unexpected error occurred: {e}")