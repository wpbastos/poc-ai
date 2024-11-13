# main.py
import streamlit as st
import logging
from pathlib import Path
from src.core.config import ConfigManager
from src.clients.langchain_client import LangChainClient
from src.clients.redis_history import RedisLangChainHistory
from src.ui.streamlit_app import StreamlitApp
from src.utils.logger import setup_logging

def setup_environment():
    """Setup application environment and logging."""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    log_file = Path("logs/app.log")
    setup_logging(
        log_file=log_file,
        log_level=logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LLM Infrastructure Analyzer...")

def initialize_clients(config):
    """Initialize application clients with error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize LangChain client
        llm_client = LangChainClient(config)
        logger.info("LangChain client initialized successfully")
        
        # Initialize Redis chat history
        chat_history = RedisLangChainHistory(config)
        logger.info("Redis chat history initialized successfully")
        
        return llm_client, chat_history
        
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

def main():
    """Main application entry point."""
    try:
        # Setup environment and logging
        setup_environment()
        logger = logging.getLogger(__name__)
        
        # Set page config
        st.set_page_config(
            page_title="LLM Infrastructure Analyzer",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load configuration
        config = ConfigManager().get_config()
        
        # Initialize clients
        llm_client, chat_history = initialize_clients(config)
        
        # Initialize and run Streamlit app
        app = StreamlitApp(config, llm_client, chat_history)
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred while running the application. Please check the logs for details.")
        if config and config.debug:
            st.exception(e)

if __name__ == "__main__":
    main()