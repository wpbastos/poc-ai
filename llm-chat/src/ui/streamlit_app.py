# src/ui/streamlit_app.py
import streamlit as st
from datetime import datetime
from src.ui.components.sidebar import Sidebar
from src.ui.components.chat import ChatComponent
from src.core.config import AppConfig
from src.clients.base import BaseLLMClient, BaseChatHistory

class StreamlitApp:
    def __init__(
        self,
        config: AppConfig,
        llm_client: BaseLLMClient,
        chat_history: BaseChatHistory
    ):
        self.config = config
        self.llm_client = llm_client
        self.chat_history = chat_history

    def run(self):
        st.title("LLM Chat Interface")

        # Initialize session state
        self._initialize_session_state()

        # Render sidebar
        sidebar = Sidebar(self.llm_client, self.chat_history)
        model, temperature, use_streaming = sidebar.render()

        # Display current session info
        if st.session_state.current_session:
            self._display_session_info()

        # Render chat interface
        chat = ChatComponent(
            self.llm_client,
            self.chat_history,
            st.session_state.current_session
        )
        chat.render(
            st.session_state.messages,
            model,
            temperature,
            use_streaming
        )

    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if "current_session" not in st.session_state:
            st.session_state.current_session = self.chat_history.create_session()

        if "messages" not in st.session_state:
            if st.session_state.current_session:
                st.session_state.messages = self.chat_history.get_session_messages(
                    st.session_state.current_session
                )
            else:
                st.session_state.messages = []

    def _display_session_info(self):
        """Display current session information."""
        session_info = self.chat_history.get_session_info(
            st.session_state.current_session
        )
        created_at = datetime.fromisoformat(
            session_info.get("created_at", datetime.now().isoformat())
        )
        st.caption(
            f"Current Session: Started {created_at.strftime('%Y-%m-%d %H:%M')} - "
            f"{session_info.get('message_count', 0)} messages"
        )