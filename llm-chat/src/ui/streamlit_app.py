# src/ui/streamlit_app.py
import streamlit as st
import logging
from datetime import datetime
from typing import Optional
from src.ui.components.sidebar import Sidebar
from src.ui.components.chat import ChatComponent
from src.core.config import AppConfig
from src.clients.langchain_client import LangChainClient
from src.clients.base import BaseChatHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

class StreamlitApp:
    def __init__(
        self,
        config: AppConfig,
        llm_client: LangChainClient,
        chat_history: BaseChatHistory
    ):
        self.config = config
        self.llm_client = llm_client
        self.chat_history = chat_history

    def run(self):
        st.title("Lucy - AI Infrastructure Analyzer 🤖")
        
        # Add welcome message
        if "welcomed" not in st.session_state:
            st.markdown("""
            ### Welcome! I'm Lucy 👋
            
            I'm an advanced AI infrastructure analyst inspired by the movie 'Lucy'. Like my namesake, 
            I possess enhanced analytical capabilities focused on understanding and optimizing cloud infrastructure.
            
            I can help you with:
            - Analyzing infrastructure configurations
            - Providing security recommendations
            - Optimizing costs
            - Improving performance
            - Ensuring compliance
            
            Feel free to ask me anything about your infrastructure!
            """)
            st.session_state.welcomed = True

        # Initialize session state
        self._initialize_session_state()

        # Render sidebar
        sidebar = Sidebar(self.llm_client, self.chat_history)
        model, temperature, use_streaming = sidebar.render()

        # Display current session info
        if st.session_state.current_session:
            self._display_session_info()

        # Display chat messages
        self._display_chat_messages()

        # Handle user input
        self._handle_user_input(model, temperature, use_streaming)

    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if "current_session" not in st.session_state:
            # Create new session if none exists
            new_session = self.chat_history.create_session()
            st.session_state.current_session = new_session
            st.session_state.messages = []
        elif "messages" not in st.session_state:
            # Load messages for existing session
            st.session_state.messages = self.chat_history.get_session_messages(
                st.session_state.current_session
            )

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

    def _display_chat_messages(self):
        """Display all messages in the current chat session."""
        for message in st.session_state.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(message.content)

    def _handle_user_input(self, model: str, temperature: float, use_streaming: bool):
        """Handle user input and generate response."""
        if prompt := st.chat_input("How can I assist you with your infrastructure today?"):
            # Add user message to UI
            with st.chat_message("user"):
                st.write(prompt)

            # Create and save user message
            user_message = HumanMessage(content=prompt)
            st.session_state.messages.append(user_message)
            self.chat_history.add_message(st.session_state.current_session, user_message)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                if use_streaming:
                    response = self._handle_streaming_response(
                        prompt, model, temperature
                    )
                else:
                    response = self._handle_standard_response(
                        prompt, model, temperature
                    )

                if response:
                    # Create and save assistant message
                    ai_message = AIMessage(content=response)
                    st.session_state.messages.append(ai_message)
                    self.chat_history.add_message(
                        st.session_state.current_session,
                        ai_message
                    )

    def _handle_streaming_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> Optional[str]:
        """Handle streaming response generation."""
        message_placeholder = st.empty()
        full_response = []

        def handle_token(token: str):
            full_response.append(token)
            message_placeholder.markdown("".join(full_response) + "▌")

        try:
            response = self.llm_client.generate_response_stream(
                prompt=prompt,
                model=model,
                temperature=temperature,
                streaming_callback=handle_token,
                session_id=st.session_state.current_session
            )

            final_response = "".join(full_response)
            message_placeholder.markdown(final_response)
            return final_response

        except Exception as e:
            st.error("Failed to get streaming response")
            logger.error(f"Streaming error: {str(e)}")
            return None

    def _handle_standard_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> Optional[str]:
        """Handle standard (non-streaming) response generation."""
        try:
            with st.spinner("Analyzing your request..."):
                response = self.llm_client.generate_response(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    session_id=st.session_state.current_session
                )
                if response:
                    st.write(response)
                    return response
                else:
                    st.error("Failed to generate response")
                    return None

        except Exception as e:
            st.error("Failed to generate response")
            logger.error(f"Response error: {str(e)}")
            return None