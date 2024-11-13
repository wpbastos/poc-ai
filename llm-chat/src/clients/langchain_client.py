# src/clients/langchain_client.py
from typing import Optional, Iterator, List, Dict, Any
import logging
import requests
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.core.exceptions import ClientConnectionError, ModelError

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, streaming_callback):
        self.streaming_callback = streaming_callback
        self.generated_text = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streaming_callback(token)
        self.generated_text.append(token)

class InMemoryMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

class LangChainClient:
    def __init__(self, config):
        """Initialize LangChain client with configuration."""
        self.config = config
        self.base_url = config.llm.host
        self.llm = None
        self.chat_model = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LangChain LLM and conversation chain."""
        try:
            self.chat_model = ChatOllama(
                base_url=self.base_url,
                model="llama2",
                temperature=self.config.llm.temperature,
                callbacks=[],
            )

            # Create system message for the assistant
            self.system_message = SystemMessage(
                content="You are a helpful AI assistant specialized in infrastructure analysis."
            )

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise ClientConnectionError(f"Failed to initialize LLM: {str(e)}")

    def _make_ollama_request(self, endpoint: str, method: str = "GET", json_data: Dict = None) -> Dict:
        """Make a request to Ollama API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=json_data, timeout=10)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error: {str(e)}")
            raise ClientConnectionError(f"Failed to connect to Ollama API: {str(e)}")

    def _get_message_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get message history for a session."""
        if not hasattr(self, '_message_histories'):
            self._message_histories = {}
        if session_id not in self._message_histories:
            self._message_histories[session_id] = InMemoryMessageHistory()
        return self._message_histories[session_id]

    def generate_response(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = None,
        max_tokens: int = None,
        session_id: str = "default"
    ) -> Optional[str]:
        """Generate a response using LangChain."""
        try:
            # Update model configuration if needed
            if (model != self.chat_model.model or 
                temperature != self.chat_model.temperature):
                self.chat_model = ChatOllama(
                    base_url=self.base_url,
                    model=model,
                    temperature=temperature or self.config.llm.temperature,
                    max_tokens=max_tokens or self.config.llm.max_tokens
                )

            # Get message history
            history = self._get_message_history(session_id)
            
            # Create messages list with system message and history
            messages = [self.system_message] + history.get_messages() + [HumanMessage(content=prompt)]

            # Generate response
            response = self.chat_model.invoke(messages)
            
            # Add messages to history
            history.add_message(HumanMessage(content=prompt))
            history.add_message(response)

            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ModelError(f"Failed to generate response: {str(e)}")

    def generate_response_stream(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = None,
        max_tokens: int = None,
        streaming_callback=None,
        session_id: str = "default"
    ) -> Iterator[str]:
        """Generate a streaming response using LangChain."""
        try:
            callback_handler = StreamingCallbackHandler(streaming_callback)
            
            # Create streaming model
            streaming_model = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature or self.config.llm.temperature,
                max_tokens=max_tokens or self.config.llm.max_tokens,
                streaming=True,
                callbacks=[callback_handler]
            )

            # Get message history
            history = self._get_message_history(session_id)
            
            # Create messages list with system message and history
            messages = [self.system_message] + history.get_messages() + [HumanMessage(content=prompt)]

            # Generate streaming response
            response = streaming_model.invoke(messages)
            
            # Add messages to history
            history.add_message(HumanMessage(content=prompt))
            history.add_message(response)

            return response.content

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            raise ModelError(f"Streaming error: {str(e)}")

    def list_models(self) -> List[Dict]:
        """List available models from Ollama."""
        try:
            # Get models list from Ollama API
            response = self._make_ollama_request("api/tags")
            
            # Extract and format model information
            models = []
            for model in response.get("models", []):
                model_info = {
                    "name": model["name"],
                    "provider": "ollama",
                    "status": "available",
                    "size": model.get("size", "unknown"),
                    "modified_at": model.get("modified_at", "unknown"),
                    "details": {
                        "digest": model.get("digest", ""),
                        "parent_digest": model.get("parent_digest", ""),
                        "modelfile": model.get("modelfile", ""),
                    }
                }
                models.append(model_info)
            
            if not models:
                logger.warning("No models found in Ollama")
                return [{"name": "llama2", "provider": "ollama", "status": "available"}]
                
            return models

        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            # Return default model if we can't get the list
            return [{"name": "llama2", "provider": "ollama", "status": "available"}]