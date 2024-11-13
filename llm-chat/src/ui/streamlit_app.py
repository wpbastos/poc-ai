# src/ui/streamlit_app.py

import streamlit as st
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from src.ui.components.sidebar import Sidebar
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

    def _add_custom_css(self):
        """Add custom CSS for tooltips and styling."""
        st.markdown("""
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 600px;
            background-color: #1E1E1E;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 12px;
            position: absolute;
            z-index: 1;
            right: 105%;
            top: -200px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85em;
            white-space: pre-wrap;
            font-family: 'Courier New', Courier, monospace;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #333;
        }

        .tooltip .tooltiptext::-webkit-scrollbar {
            width: 8px;
        }

        .tooltip .tooltiptext::-webkit-scrollbar-track {
            background: #2D2D2D;
            border-radius: 4px;
        }

        .tooltip .tooltiptext::-webkit-scrollbar-thumb {
            background: #666;
            border-radius: 4px;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        .prompt-metadata {
            padding-bottom: 8px;
            margin-bottom: 8px;
            border-bottom: 1px solid #333;
            color: #888;
        }

        .prompt-content {
            color: #fff;
        }

        div[data-testid="stToolbar"] {
            display: none;
        }

        .stMarkdown {
            min-height: auto;
        }
        </style>
        """, unsafe_allow_html=True)

    def _initialize_session_state(self):
        """Initialize Streamlit session state without creating a new session."""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_session = None
            st.session_state.messages = []
            logger.info("Initialized new session state")

    def run(self):
        """Run the Streamlit application."""
        # Add custom CSS for tooltips
        self._add_custom_css()
        
        st.title("Lucy - AI Infrastructure Analyzer 🤖")
        
        # Initialize session state without creating a new session
        self._initialize_session_state()
        
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

        # Render sidebar
        sidebar = Sidebar(self.llm_client, self.chat_history)
        model, temperature, use_streaming = sidebar.render()

        # Display main chat interface
        if st.session_state.current_session:
            # Show current session info and messages
            self._display_session_info()
            self._display_chat_messages()
            
            # Only show input box when session is active
            if prompt := st.chat_input("How can I assist you with your infrastructure today?"):
                self._handle_message(prompt, model, temperature, use_streaming)
        else:
            # Show message prompting to start a new chat
            st.info("👈 Click 'New Chat' in the sidebar to start a conversation!")

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

    def _format_tooltip_content(self, message: HumanMessage) -> str:
        """Format tooltip content with metadata and full prompt."""
        if not hasattr(message, 'prompt_info'):
            return ""
            
        info = message.prompt_info
        metadata = (
            f"Model: {info.get('model', 'N/A')}\n"
            f"Temperature: {info.get('temperature', 'N/A')}\n"
            f"Response Time: {info.get('response_time', 0):.2f}s\n"
        )
        
        full_prompt = info.get('full_prompt', 'N/A')
        
        return f"""<div class="prompt-metadata">{metadata}</div>
                  <div class="prompt-content">{full_prompt}</div>"""

    def _display_chat_messages(self):
        """Display all messages in the current chat session."""
        messages_to_display = []
        
        # First pass: collect and update messages
        for idx, message in enumerate(st.session_state.messages):
            if isinstance(message, HumanMessage):
                next_idx = idx + 1
                if next_idx < len(st.session_state.messages):
                    next_message = st.session_state.messages[next_idx]
                    if isinstance(next_message, AIMessage) and not hasattr(message, 'prompt_info'):
                        # Add prompt info if missing
                        message.prompt_info = {
                            'model': st.session_state.get('last_model', 'N/A'),
                            'temperature': st.session_state.get('last_temperature', 'N/A'),
                            'response_time': st.session_state.get('last_response_time', 0),
                            'full_prompt': st.session_state.get('last_full_prompt', 'N/A')
                        }
            messages_to_display.append(message)

        # Second pass: display messages
        for message in messages_to_display:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            
            if role == "user":
                cols = st.columns([85, 10, 5])
                
                with cols[0]:
                    with st.chat_message(role):
                        st.write(message.content)
                
                with cols[1]:
                    if hasattr(message, 'prompt_info'):
                        response_time = message.prompt_info.get('response_time', 0)
                        st.markdown(
                            f"<div style='text-align: right; padding-top: 10px; color: #666; font-size: 0.9em;'>"
                            f"{response_time:.2f}s</div>", 
                            unsafe_allow_html=True
                        )
                
                with cols[2]:
                    if hasattr(message, 'prompt_info'):
                        tooltip_content = self._format_tooltip_content(message)
                        st.markdown(f"""
                            <div class="tooltip" style="text-align: right; padding-top: 10px;">
                                <span style="font-size: 1.2em;">ℹ️</span>
                                <span class="tooltiptext">
                                    {tooltip_content}
                                </span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                with st.chat_message(role):
                    st.write(message.content)

    def _handle_message(self, prompt: str, model: str, temperature: float, use_streaming: bool):
        """Handle new message in chat."""
        if not prompt.strip():
            return

        # Create user message
        user_message = HumanMessage(content=prompt)
        
        # Add user message to UI
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            if use_streaming:
                response, response_time, full_prompt = self._handle_streaming_response(
                    prompt, model, temperature
                )
            else:
                response, response_time, full_prompt = self._handle_standard_response(
                    prompt, model, temperature
                )

            if response and response.strip():
                # Store current message info in session state
                st.session_state.last_model = model
                st.session_state.last_temperature = temperature
                st.session_state.last_response_time = response_time
                st.session_state.last_full_prompt = full_prompt

                # Add prompt info to user message
                user_message.prompt_info = {
                    'model': model,
                    'temperature': temperature,
                    'prompt': prompt,
                    'response_time': response_time,
                    'full_prompt': full_prompt
                }
                
                # Save messages only if we have a valid exchange
                st.session_state.messages.append(user_message)
                self.chat_history.add_message(st.session_state.current_session, user_message)
                
                # Create and save assistant message
                ai_message = AIMessage(content=response)
                st.session_state.messages.append(ai_message)
                self.chat_history.add_message(st.session_state.current_session, ai_message)

                # Generate chat name for first message
                if len(st.session_state.messages) == 2:
                    try:
                        chat_name = self.llm_client.generate_chat_name(prompt, response)
                        if chat_name and chat_name.strip():
                            self.chat_history.update_chat_name(
                                st.session_state.current_session, 
                                chat_name.strip()
                            )
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error generating chat name: {str(e)}")
                        
                # Force a rerun to update the display immediately
                st.rerun()

    def _handle_streaming_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> tuple[Optional[str], float, Optional[str]]:
        """Handle streaming response generation."""
        message_placeholder = st.empty()
        full_response = []
        start_time = time.time()

        def handle_token(token: str):
            full_response.append(token)
            message_placeholder.markdown("".join(full_response) + "▌")

        try:
            response, full_prompt = self.llm_client.generate_response_stream(
                prompt=prompt,
                model=model,
                temperature=temperature,
                streaming_callback=handle_token,
                session_id=st.session_state.current_session
            )

            final_response = "".join(full_response)
            message_placeholder.markdown(final_response)
            response_time = time.time() - start_time
            return final_response, response_time, full_prompt

        except Exception as e:
            st.error("Failed to get streaming response")
            logger.error(f"Streaming error: {str(e)}")
            return None, 0, None

    def _handle_standard_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> tuple[Optional[str], float, Optional[str]]:
        """Handle standard (non-streaming) response generation."""
        try:
            start_time = time.time()
            with st.spinner("Analyzing your request..."):
                response, full_prompt = self.llm_client.generate_response(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    session_id=st.session_state.current_session
                )
            response_time = time.time() - start_time
            
            if response:
                st.write(response)
                return response, response_time, full_prompt
            else:
                st.error("Failed to generate response")
                return None, 0, None

        except Exception as e:
            st.error("Failed to generate response")
            logger.error(f"Response error: {str(e)}")
            return None, 0, None