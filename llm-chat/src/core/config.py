# src/core/config.py
from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass
import os
from dotenv import load_dotenv

@dataclass
class LLMConfig:
    host: str
    temperature: float
    max_tokens: int

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
    def __init__(self, config_path: str = None):
        load_dotenv()
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        return AppConfig(
            llm=LLMConfig(
                host=os.getenv('OLLAMA_HOST', config_data['llm']['host']),
                temperature=config_data['llm']['temperature'],
                max_tokens=config_data['llm']['max_tokens']
            ),
            redis=RedisConfig(
                host=os.getenv('REDIS_HOST', config_data['redis']['host']),
                port=int(os.getenv('REDIS_PORT', config_data['redis']['port'])),
                db=config_data['redis']['db'],
                max_retries=config_data['redis']['max_retries'],
                retry_interval=config_data['redis']['retry_interval']
            ),
            debug=config_data.get('debug', False)
        )

    def get_config(self) -> AppConfig:
        return self.config