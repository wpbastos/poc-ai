# src/ui/components/chat.py
import streamlit as st
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.clients.langchain_client import LangChainClient
from src.clients.base import BaseChatHistory

class ChatComponent:
    def __init__(
        self,
        llm_client: LangChainClient,
        chat_history: BaseChatHistory,
        session_id: str
    ):
        self.llm_client = llm_client
        self.chat_history = chat_history
        self.session_id = session_id

    def render(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        use_streaming: bool
    ) -> None:
        """Render the chat interface."""
        # Display messages
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(message.content)

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
        user_message = HumanMessage(content=prompt)
        self.chat_history.add_message(self.session_id, user_message)

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
        message_placeholder = st.empty()
        full_response = []

        def handle_token(token: str):
            full_response.append(token)
            message_placeholder.markdown("".join(full_response) + "▌")

        response = self.llm_client.generate_response_stream(
            prompt=prompt,
            model=model,
            temperature=temperature,
            streaming_callback=handle_token
        )

        if response:
            final_response = "".join(full_response)
            message_placeholder.markdown(final_response)
            ai_message = AIMessage(content=final_response)
            self.chat_history.add_message(self.session_id, ai_message)
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
                ai_message = AIMessage(content=response)
                self.chat_history.add_message(self.session_id, ai_message)
            else:
                st.error("Failed to get response")