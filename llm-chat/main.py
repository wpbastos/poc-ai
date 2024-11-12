# main.py
import streamlit as st
from src.llm.ollama_client import OllamaClient
from src.utils.redis_history import RedisChatHistory
import yaml
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_env():
    logger.info("Environment Variables:")
    logger.info(f"OLLAMA_HOST: {os.getenv('OLLAMA_HOST')}")
    logger.info(f"REDIS_HOST: {os.getenv('REDIS_HOST')}")
    logger.info(f"REDIS_PORT: {os.getenv('REDIS_PORT')}")

# Initialize clients
@st.cache_resource
def init_clients():
    ollama_host = os.getenv('OLLAMA_HOST')
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    # Initialize Ollama client with explicit host
    ollama = OllamaClient(base_url=ollama_host)
    logger.info(f"Initialized Ollama client with host: {ollama_host}")
    
    try:
        history = RedisChatHistory(host=redis_host, port=redis_port)
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        return ollama, history
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        st.error("Failed to connect to Redis. Please check your configuration.")
        return ollama, None

def format_session_info(session):
    """Format session information for display."""
    created_at = datetime.fromisoformat(session["created_at"])
    return f"{session['id'].split(':')[1]} - {created_at.strftime('%Y-%m-%d %H:%M')} ({session['message_count']} messages)"

debug_env()

def main():
    st.title("LLM Chat Interface")
    
    # Initialize components
    ollama, history = init_clients()
    
    if not history:
        st.error("Chat history is not available - Redis connection failed")
        return

    # Sidebar - Connection Status and Controls
    with st.sidebar:
        st.write("Connection Status:")
        models = ollama.list_models()
        if models:
            st.success("Connected to Ollama")
            st.success("Connected to Redis")
        else:
            st.error("Not connected to Ollama")
            st.stop()

        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            [model["name"] for model in models]
        )

        # Temperature control
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )

        # Streaming option
        use_streaming = st.toggle("Use streaming responses", value=True)

        st.write("---")
        st.write("Chat Sessions")

        # New chat button
        if st.button("New Chat"):
            session_id = history.create_session()
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.rerun()

        # Clear all sessions
        if st.button("Clear All Sessions", type="secondary"):
            if st.session_state.get("confirm_clear", False):
                if history.clear_all_sessions():
                    st.success("All sessions cleared")
                    st.session_state.current_session = None
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all sessions")

        # List and select existing sessions
        sessions = history.list_sessions()
        if sessions:
            selected_session = st.selectbox(
                "Load Previous Chat",
                sessions,
                format_func=format_session_info
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key="load_btn"):
                    st.session_state.current_session = selected_session["id"]
                    st.session_state.messages = history.get_session_messages(selected_session["id"])
                    st.rerun()
            
            with col2:
                if st.button("Delete", key="delete_btn", type="secondary"):
                    if history.delete_session(selected_session["id"]):
                        if selected_session["id"] == st.session_state.get('current_session'):
                            st.session_state.current_session = None
                            st.session_state.messages = []
                        st.success("Session deleted")
                        st.rerun()
                    else:
                        st.error("Error deleting session")

    # Initialize or load session
    if "current_session" not in st.session_state:
        st.session_state.current_session = history.create_session()

    if "messages" not in st.session_state:
        if st.session_state.current_session:
            st.session_state.messages = history.get_session_messages(st.session_state.current_session)
        else:
            st.session_state.messages = []

    # Display current session info
    if st.session_state.current_session:
        session_info = history.get_session_info(st.session_state.current_session)
        created_at = datetime.fromisoformat(session_info.get("created_at", datetime.now().isoformat()))
        st.caption(f"Current Session: Started {created_at.strftime('%Y-%m-%d %H:%M')} - {session_info.get('message_count', 0)} messages")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Send a message..."):
        # Add user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Save user message
        history.add_message(st.session_state.current_session, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            if use_streaming:
                # Initialize empty message for streaming
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
                        message_placeholder.markdown(full_response + "▌")
                
                # Final update without cursor
                if full_response:
                    message_placeholder.markdown(full_response)
                    # Save assistant response
                    history.add_message(st.session_state.current_session, "assistant", full_response)
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
                        # Save assistant response
                        history.add_message(st.session_state.current_session, "assistant", response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get response from Ollama")

if __name__ == "__main__":
    main()