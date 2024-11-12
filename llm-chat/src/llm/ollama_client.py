# src/llm/ollama_client.py
import requests
import json
import logging
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or "http://localhost:11434"
        self.api_endpoint = f"{self.base_url}/api/generate"
        
    def generate_response(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """Generate a response from Ollama model."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
                return None
                
            return response.json().get("response", "")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
            
    def generate_response_stream(
        self,
        prompt: str,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Iterator[str]:
        """
        Generate a streaming response from Ollama model.
        Returns an iterator that yields response chunks.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
                yield None
                return
                
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing streaming response: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield None
            
    def list_models(self) -> Optional[list]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            
            if response.status_code != 200:
                logger.error(f"Error listing models: {response.text}")
                return None
                
            models = response.json().get("models", [])
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return None