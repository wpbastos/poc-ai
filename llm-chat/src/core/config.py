# src/core/config.py
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import os
import logging
from dotenv import load_dotenv

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    host: str
    temperature: float
    max_tokens: int
    model_name: str

@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    max_retries: int
    retry_interval: int

@dataclass
class AppConfig:
    llm: LLMConfig
    redis: RedisConfig
    debug: bool

class ConfigManager:
    def __init__(self):
        """Initialize configuration manager."""
        # Load environment variables
        load_dotenv()
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from environment variables."""
        # Get temperature with proper type conversion and error handling
        try:
            temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.5'))
        except ValueError:
            logger.warning("Invalid OLLAMA_TEMPERATURE value, using default 0.5")
            temperature = 0.5

        return AppConfig(
            llm=LLMConfig(
                host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                model_name=os.getenv('OLLAMA_MODEL', 'llama2:3b'),
                temperature=temperature,
                max_tokens=int(os.getenv('OLLAMA_MAX_TOKENS', '2048'))
            ),
            redis=RedisConfig(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                db=int(os.getenv('REDIS_DB', '0')),
                max_retries=int(os.getenv('REDIS_MAX_RETRIES', '3')),
                retry_interval=int(os.getenv('REDIS_RETRY_INTERVAL', '1'))
            ),
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        )

    def get_config(self) -> AppConfig:
        """Get the loaded configuration."""
        return self.config