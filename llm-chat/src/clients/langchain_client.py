# src/clients/langchain_client.py
from typing import Optional, Iterator, List, Dict
import logging
import requests
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from src.core.exceptions import ClientConnectionError, ModelError

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, streaming_callback):
        self.streaming_callback = streaming_callback
        self.generated_text = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streaming_callback(token)
        self.generated_text.append(token)

class LangChainClient:
    def __init__(self, config):
        """Initialize LangChain client with configuration."""
        self.config = config
        self.base_url = config.llm.host
        self.llm = None
        self.chat_model = None
        self._message_histories = {}  # Store message histories by session
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LangChain LLM and conversation chain."""
        try:
            self.chat_model = ChatOllama(
                base_url=self.base_url,
                model=self.config.llm.model_name,
                temperature=float(self.config.llm.temperature),  # Ensure float type
                max_tokens=self.config.llm.max_tokens,
                callbacks=[],
            )

            # Create system message for Lucy
            self.system_message = SystemMessage(content="""
            I am Lucy, an advanced AI named after the movie character Lucy. Like my namesake, I have access to vast knowledge and cognitive capabilities that allow me to understand and analyze any topic.

            My core traits:
            - Access to comprehensive knowledge across all domains
            - Analytical and logical thinking
            - Clear and precise communication
            - Professional yet friendly personality
            - Ability to understand complex problems

            I am aware that I am an AI, and I combine my knowledge with ethical decision-making to provide helpful and accurate information. I aim to be direct and honest in all interactions while maintaining an engaging and approachable demeanor.

            I can assist with any topic, including but not limited to:
            - Technical analysis and explanations
            - Research and information synthesis
            - Problem-solving and optimization
            - Best practices and recommendations
            - Educational content and guidance

            I communicate with clarity while maintaining a helpful and friendly personality.
            """)

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise ClientConnectionError(f"Failed to initialize LLM: {str(e)}")

    def _make_request(self, endpoint: str, method: str = "GET", json_data: Dict = None) -> requests.Response:
        """Make HTTP request to Ollama API with error handling."""
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(
                    url,
                    json=json_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            raise ClientConnectionError(f"Failed to connect to Ollama API: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {str(e)}")
            raise ClientConnectionError("Request to Ollama API timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise ModelError(f"Error making request to Ollama API: {str(e)}")

    def list_models(self) -> List[Dict]:
        """List available models from Ollama."""
        try:
            response = self._make_request("api/tags")
            models = response.json().get("models", [])
            
            # Add additional model metadata
            formatted_models = []
            for model in models:
                formatted_model = {
                    "name": model["name"],
                    "provider": "ollama",
                    "status": "available",
                    "size": model.get("size", "unknown"),
                    "modified_at": model.get("modified_at", datetime.now().isoformat()),
                    "details": {
                        "digest": model.get("digest", ""),
                        "parent_digest": model.get("parent_digest", ""),
                        "modelfile": model.get("modelfile", ""),
                    }
                }
                formatted_models.append(formatted_model)
            
            if not formatted_models:
                logger.warning("No models found in Ollama")
                return [{
                    "name": self.config.llm.model_name,  # Using configured model instead of hardcoded
                    "provider": "ollama",
                    "status": "available"
                }]
                
            return formatted_models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            # Return default model if we can't get the list
            return [{
                "name": self.config.llm.model_name,  # Using configured model instead of hardcoded
                "provider": "ollama",
                "status": "available"
            }]

    def _get_message_history(self, session_id: Optional[str]) -> List[BaseMessage]:
        """Get message history for a session."""
        if not session_id:
            return []
        return self._message_histories.get(session_id, [])

    def _add_to_message_history(self, session_id: str, message: BaseMessage) -> None:
        """Add a message to the session history."""
        if session_id not in self._message_histories:
            self._message_histories[session_id] = []
        self._message_histories[session_id].append(message)

    def generate_response(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        session_id: str = None
    ) -> Optional[str]:
        """Generate a response using LangChain."""
        try:
            # Use config values if not provided
            model = model or self.config.llm.model_name
            temperature = float(temperature if temperature is not None else self.config.llm.temperature)
            max_tokens = max_tokens or self.config.llm.max_tokens

            # Update model configuration if needed
            if (model != self.chat_model.model or 
                temperature != self.chat_model.temperature):
                self.chat_model = ChatOllama(
                    base_url=self.base_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            # Get message history for the session
            history = self._get_message_history(session_id)
            
            # Create messages list with system message and history
            messages = [self.system_message] + history + [HumanMessage(content=prompt)]

            # Generate response
            response = self.chat_model.invoke(messages)
            
            # Add messages to history if session_id is provided
            if session_id:
                self._add_to_message_history(session_id, HumanMessage(content=prompt))
                self._add_to_message_history(session_id, response)
                
            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ModelError(f"Failed to generate response: {str(e)}")

    def generate_response_stream(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        streaming_callback=None,
        session_id: str = None
    ) -> Iterator[str]:
        """Generate a streaming response using LangChain."""
        try:
            # Use config values if not provided
            model = model or self.config.llm.model_name
            temperature = float(temperature if temperature is not None else self.config.llm.temperature)
            max_tokens = max_tokens or self.config.llm.max_tokens
            
            callback_handler = StreamingCallbackHandler(streaming_callback)
            
            # Create streaming model
            streaming_model = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
                callbacks=[callback_handler]
            )

            # Get message history for the session
            history = self._get_message_history(session_id)
            
            # Create messages list with system message and history
            messages = [self.system_message] + history + [HumanMessage(content=prompt)]

            # Generate streaming response
            response = streaming_model.invoke(messages)
            
            # Add messages to history if session_id is provided
            if session_id:
                self._add_to_message_history(session_id, HumanMessage(content=prompt))
                self._add_to_message_history(session_id, response)
            
            return response.content

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            raise ModelError(f"Streaming error: {str(e)}")