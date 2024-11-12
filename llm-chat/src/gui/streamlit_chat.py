# src/gui/streamlit_chat.py
import streamlit as st
from typing import Callable
import yaml

class StreamlitChat:
    def __init__(self, config_path: str, send_callback: Callable):
        self.config = self._load_config(config_path)
        self.send_callback = send_callback
        self._init_ui()
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _init_ui(self):
        st.title("LLM Chat Interface")
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Temperature slider in sidebar
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=self.config['llm']['temperature'],
            step=0.05
        )
        
        # Chat input
        if prompt := st.chat_input("What's on your mind?"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Get response
            response = self.send_callback(prompt, temperature)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)