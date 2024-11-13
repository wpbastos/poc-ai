# src/clients/base.py
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any, List
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage

class BaseChatHistory(ABC):
    @abstractmethod
    def create_session(self) -> str:
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        pass

    @abstractmethod
    def get_session_messages(self, session_id: str) -> List[BaseMessage]:
        pass

    @abstractmethod
    def get_session_info(self, session_id: str) -> Dict:
        pass

    @abstractmethod
    def list_sessions(self) -> List[Dict]:
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def clear_all_sessions(self) -> bool:
        pass