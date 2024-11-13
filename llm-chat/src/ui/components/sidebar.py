# src/ui/components/sidebar.py
import streamlit as st
from typing import Tuple, Any, List, Dict
from datetime import datetime
import logging
from src.utils.formatting import format_session_info
from src.clients.base import BaseLLMClient, BaseChatHistory

logger = logging.getLogger(__name__)

class Sidebar:
    def __init__(self, ollama_client: BaseLLMClient, chat_history: BaseChatHistory):
        """
        Initialize sidebar component.
        
        Args:
            ollama_client: LLM client instance
            chat_history: Chat history client instance
        """
        self.ollama_client = ollama_client
        self.chat_history = chat_history
        
    def render(self) -> Tuple[str, float, bool]:
        """
        Render the sidebar components.
        
        Returns:
            Tuple of (selected_model, temperature, use_streaming)
        """
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
            models = self.ollama_client.list_models()
            if models:
                st.success("✓ Connected to Ollama")
                st.success("✓ Models Available")
            else:
                st.error("× No Models Available")
                
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            st.error("× Failed to Connect to Ollama")
            st.stop()
            
    def _render_model_selection(self) -> str:
        """
        Render model selection dropdown.
        
        Returns:
            Selected model name
        """
        st.write("### Model Settings")
        
        try:
            models = self.ollama_client.list_models()
            model_names = [model["name"] for model in models]
            
            selected_model = st.selectbox(
                "Select Model",
                options=model_names,
                help="Choose the AI model to use for responses"
            )
            
            # Display model info if available
            if selected_model:
                model_info = self.ollama_client.get_model_info(selected_model)
                if model_info:
                    st.caption(f"Model Size: {model_info.get('size', 'Unknown')}")
                    st.caption(f"Last Updated: {model_info.get('last_updated', 'Unknown')}")
                    
            return selected_model
            
        except Exception as e:
            logger.error(f"Error rendering model selection: {str(e)}")
            st.error("Failed to load models")
            return "llama2"  # Default fallback
            
    def _render_temperature_control(self) -> float:
        """
        Render temperature slider control.
        
        Returns:
            Selected temperature value
        """
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused"
        )
        
        # Display temperature description
        if temperature < 0.3:
            st.caption("🎯 More focused and deterministic responses")
        elif temperature > 0.7:
            st.caption("🎲 More creative and varied responses")
        else:
            st.caption("⚖️ Balanced between focus and creativity")
            
        return temperature
        
    def _render_streaming_toggle(self) -> bool:
        """
        Render streaming response toggle.
        
        Returns:
            Boolean indicating if streaming is enabled
        """
        return st.toggle(
            "Enable Streaming",
            value=True,
            help="Show responses as they are generated"
        )
        
    def _render_session_management(self) -> None:
        """Render session management controls."""
        st.write("### Chat Sessions")
        
        # New chat button
        if st.button("New Chat", use_container_width=True):
            session_id = self.chat_history.create_session()
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.rerun()
            
        # Session list
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
                if st.button("Delete", use_container_width=True, type="secondary"):
                    if st.session_state.get("confirm_delete") == selected_session["id"]:
                        if self.chat_history.delete_session(selected_session["id"]):
                            if selected_session["id"] == st.session_state.get('current_session'):
                                st.session_state.current_session = None
                                st.session_state.messages = []
                            st.success("Session deleted")
                            st.session_state.confirm_delete = None
                            st.rerun()
                        else:
                            st.error("Error deleting session")
                    else:
                        st.session_state.confirm_delete = selected_session["id"]
                        st.warning("Click again to confirm deletion")
                        
        # Clear all sessions
        if st.button("Clear All Sessions", type="secondary", use_container_width=True):
            if st.session_state.get("confirm_clear", False):
                if self.chat_history.clear_all_sessions():
                    st.success("All sessions cleared")
                    st.session_state.current_session = None
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.error("Error clearing sessions")
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all sessions")