# src/ui/components/chat.py
import streamlit as st
from typing import Optional, Callable
from src.clients.base import BaseLLMClient, BaseChatHistory

class ChatComponent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        chat_history: BaseChatHistory,
        session_id: str
    ):
        self.llm_client = llm_client
        self.chat_history = chat_history
        self.session_id = session_id

    def render(
        self,
        messages: list,
        model: str,
        temperature: float,
        use_streaming: bool
    ) -> None:
        """Render the chat interface."""
        # Display messages
        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Send a message..."):
            self._handle_user_input(
                prompt,
                model,
                temperature,
                use_streaming
            )

    def _handle_user_input(
        self,
        prompt: str,
        model: str,
        temperature: float,
        use_streaming: bool
    ) -> None:
        """Handle user input and generate response."""
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Save user message
        self.chat_history.add_message(self.session_id, "user", prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            if use_streaming:
                self._handle_streaming_response(prompt, model, temperature)
            else:
                self._handle_standard_response(prompt, model, temperature)

    def _handle_streaming_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> None:
        """Handle streaming response generation."""
        full_response = ""
        message_placeholder = st.empty()

        for response_chunk in self.llm_client.generate_response_stream(
            prompt=prompt,
            model=model,
            temperature=temperature
        ):
            if response_chunk:
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")

        if full_response:
            message_placeholder.markdown(full_response)
            self.chat_history.add_message(
                self.session_id,
                "assistant",
                full_response
            )
        else:
            st.error("Failed to get streaming response")

    def _handle_standard_response(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> None:
        """Handle standard (non-streaming) response generation."""
        with st.spinner("Generating response..."):
            response = self.llm_client.generate_response(
                prompt=prompt,
                model=model,
                temperature=temperature
            )

            if response:
                st.write(response)
                self.chat_history.add_message(
                    self.session_id,
                    "assistant",
                    response
                )
            else:
                st.error("Failed to get response")