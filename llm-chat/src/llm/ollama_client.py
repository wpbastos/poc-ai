# src/llm/ollama_client.py
import requests
import json
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('OLLAMA_HOST', 'http://172.16.12.102:11434')
        self.api_endpoint = f"{self.base_url}/api/generate"
        logger.info(f"Initialized Ollama client with base URL: {self.base_url}")
        
    def generate_response(
        self,
        prompt: str,
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate a response from Ollama model with stream handling.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False  # Set to False to get a single response
            }
            
            logger.info(f"Sending request to {self.api_endpoint}")
            logger.info(f"Payload: {payload}")
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
                return None
            
            try:
                response_data = response.json()
                return response_data.get("response", "")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                # Try to extract response from streaming data
                full_response = ""
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                        except json.JSONDecodeError:
                            continue
                return full_response if full_response else None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
            
    def generate_response_stream(
        self,
        prompt: str,
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Generate a streaming response from Ollama model.
        Returns a generator that yields response chunks.
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
            logger.info(f"Available models: {models}")
            return models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return None