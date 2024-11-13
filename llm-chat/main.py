# main.py
import streamlit as st
from src.core.config import ConfigManager
from src.clients.ollama import OllamaClient
from src.clients.redis import RedisChatHistory
from src.ui.streamlit_app import StreamlitApp
from src.utils.logger import setup_logging

def main():
    # Setup logging
    setup_logging()

    # Load configuration
    config = ConfigManager().get_config()

    # Initialize clients
    ollama_client = OllamaClient(config)
    chat_history = RedisChatHistory(config)

    # Initialize and run Streamlit app
    app = StreamlitApp(config, ollama_client, chat_history)
    app.run()

if __name__ == "__main__":
    main()