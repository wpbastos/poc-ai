# main.py
import streamlit as st
from src.llm.ollama_client import OllamaClient
import yaml
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    try:
        with open("config/settings.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

# Initialize Ollama client
@st.cache_resource
def init_ollama():
    client = OllamaClient()
    # Test connection
    models = client.list_models()
    if not models:
        st.error("Could not connect to Ollama server. Please check your connection.")
    return client

def main():
    st.title("LLM Chat Interface")
    
    # Initialize components
    config = load_config()
    ollama = init_ollama()
    
    # Add connection status indicator
    with st.sidebar:
        st.write("Connection Status:")
        models = ollama.list_models()
        if models:
            st.success("Connected to Ollama")
            st.write("Available Models:")
            for model in models:
                st.write(f"- {model['name']}")
        else:
            st.error("Not connected to Ollama")
            st.stop()
    
    # Model selection in sidebar
    if models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            [model["name"] for model in models]
        )
    else:
        selected_model = "llama3.2:3b"
    
    # Temperature control
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    # Use streaming option
    use_streaming = st.sidebar.checkbox("Use streaming responses", value=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Send a message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            if use_streaming:
                # Initialize an empty message
                full_response = ""
                message_placeholder = st.empty()
                
                # Stream the response
                for response_chunk in ollama.generate_response_stream(
                    prompt=prompt,
                    model=selected_model,
                    temperature=temperature
                ):
                    if response_chunk:
                        full_response += response_chunk
                        # Update the message in real-time
                        message_placeholder.write(full_response)
                
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Failed to get streaming response from Ollama")
            else:
                # Use non-streaming response
                with st.spinner("Generating response..."):
                    response = ollama.generate_response(
                        prompt=prompt,
                        model=selected_model,
                        temperature=temperature
                    )
                    
                    if response:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get response from Ollama")

if __name__ == "__main__":
    main()