# src/clients/ollama.py
from typing import Optional, Iterator, List, Dict
import requests
import json
import logging
from datetime import datetime
from src.core.exceptions import ClientConnectionError, ModelError
from src.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)

class OllamaClient(BaseLLMClient):
    def __init__(self, config):
        """
        Initialize Ollama client with configuration.
        
        Args:
            config: Application configuration object containing LLM settings
        """
        self.base_url = config.llm.host
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.model_endpoint = f"{self.base_url}/api/tags"
        self.default_temp = config.llm.temperature
        self.default_max_tokens = config.llm.max_tokens
        
    def _make_request(self, endpoint: str, method: str = "GET", json_data: Dict = None) -> requests.Response:
        """
        Make HTTP request to Ollama API with error handling.
        
        Args:
            endpoint: API endpoint
            method: HTTP method (GET/POST)
            json_data: Request payload for POST requests
            
        Returns:
            Response object from requests
            
        Raises:
            ClientConnectionError: For connection issues
            ModelError: For API errors
        """
        try:
            if method == "GET":
                response = requests.get(
                    endpoint,
                    timeout=10
                )
            else:
                response = requests.post(
                    endpoint,
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
        
    def generate_response(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """
        Generate a response from Ollama model.
        
        Args:
            prompt: Input text prompt
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text or None if failed
            
        Raises:
            ModelError: For generation errors
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature or self.default_temp,
                "max_tokens": max_tokens or self.default_max_tokens,
                "stream": False
            }
            
            response = self._make_request(self.api_endpoint, "POST", payload)
            return response.json().get("response", "")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ModelError(f"Failed to generate response: {str(e)}")
            
    def generate_response_stream(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = None,
        max_tokens: int = None
    ) -> Iterator[str]:
        """
        Generate a streaming response from Ollama model.
        
        Args:
            prompt: Input text prompt
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response text chunks
            
        Raises:
            ModelError: For generation errors
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature or self.default_temp,
                "max_tokens": max_tokens or self.default_max_tokens,
                "stream": True
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
                raise ModelError(f"Error response from API: {response.text}")
                
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing streaming response: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            raise ModelError(f"Streaming error: {str(e)}")
            
    def list_models(self) -> List[Dict]:
        """
        List available models from Ollama.
        
        Returns:
            List of model information dictionaries
            
        Raises:
            ModelError: For API errors
        """
        try:
            response = self._make_request(self.model_endpoint)
            models = response.json().get("models", [])
            
            # Add additional model metadata
            for model in models:
                model["last_updated"] = datetime.now().isoformat()
                model["status"] = "available"
                
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise ModelError(f"Failed to list models: {str(e)}")
            
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            models = self.list_models()
            for model in models:
                if model["name"] == model_name:
                    return model
            return None
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None