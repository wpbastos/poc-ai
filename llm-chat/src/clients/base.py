# src/clients/base.py
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        pass

    @abstractmethod
    def generate_response_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        pass

    @abstractmethod
    def list_models(self) -> Optional[list]:
        pass

class BaseChatHistory(ABC):
    @abstractmethod
    def create_session(self) -> str:
        pass

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str):
        pass

    @abstractmethod
    def get_session_messages(self, session_id: str) -> list:
        pass