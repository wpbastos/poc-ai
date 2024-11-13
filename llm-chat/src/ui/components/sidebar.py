# src/ui/components/sidebar.py
import streamlit as st
from typing import Tuple
import logging
from src.utils.formatting import format_session_info
from src.clients.langchain_client import LangChainClient
from src.clients.base import BaseChatHistory

logger = logging.getLogger(__name__)

class Sidebar:
    def __init__(self, llm_client: LangChainClient, chat_history: BaseChatHistory):
        self.llm_client = llm_client
        self.chat_history = chat_history
        self.config = llm_client.config

    def render(self) -> Tuple[str, float, bool]:
        """Render the sidebar components."""
        with st.sidebar:
            # Connection status
            self._render_connection_status()
            
            # Model selection and parameters
            model = self._render_model_selection()
            temperature = self._render_temperature_control()
            streaming = self._render_streaming_toggle()
            
            st.divider()
            
            # Session management
            self._render_session_management()
            
            return model, temperature, streaming

    def _render_connection_status(self) -> None:
        """Render connection status indicators."""
        st.write("### Connection Status")
        
        try:
            models = self.llm_client.list_models()
            if models:
                st.success("✓ Connected to LLM")
                st.success("✓ Models Available")
            else:
                st.error("× No Models Available")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            st.error("× Failed to Connect to LLM")
            st.stop()

    def _render_model_selection(self) -> str:
        """Render model selection dropdown."""
        st.write("### Model Settings")
        
        try:
            models = self.llm_client.list_models()
            model_names = [model["name"] for model in models]
            
            # Use configured model as default selection
            default_index = model_names.index(self.config.llm.model_name) if self.config.llm.model_name in model_names else 0
            
            return st.selectbox(
                "Select Model",
                options=model_names,
                index=default_index,
                help="Choose the AI model to use for responses"
            )
        except Exception as e:
            logger.error(f"Error rendering model selection: {str(e)}")
            st.error("Failed to load models")
            return self.config.llm.model_name

    def _render_temperature_control(self) -> float:
        """Render temperature slider control."""
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=self.config.llm.temperature,  # Use configured temperature
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused"
        )
        
        if temperature < 0.3:
            st.caption("🎯 More focused and deterministic responses")
        elif temperature > 0.7:
            st.caption("🎲 More creative and varied responses")
        else:
            st.caption("⚖️ Balanced between focus and creativity")
            
        return temperature

    def _render_streaming_toggle(self) -> bool:
        """Render streaming response toggle."""
        return st.toggle(
            "Enable Streaming",
            value=True,
            help="Show responses as they are generated"
        )

    def _render_session_management(self) -> None:
        """Render session management controls."""
        st.write("### Chat Sessions")
        
        if st.button("New Chat", use_container_width=True):
            session_id = self.chat_history.create_session()
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.rerun()

        sessions = self.chat_history.list_sessions()
        if sessions:
            st.write("Previous Sessions:")
            selected_session = st.selectbox(
                "Load Session",
                sessions,
                format_func=format_session_info,
                label_visibility="collapsed"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", use_container_width=True):
                    st.session_state.current_session = selected_session["id"]
                    st.session_state.messages = self.chat_history.get_session_messages(
                        selected_session["id"]
                    )
                    st.rerun()

            with col2:
                if st.button("Delete", use_container_width=True):
                    if self.chat_history.delete_session(selected_session["id"]):
                        if selected_session["id"] == st.session_state.current_session:
                            st.session_state.current_session = None
                            st.session_state.messages = []
                        st.success("Session deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete session")