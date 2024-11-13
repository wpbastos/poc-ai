# src/clients/redis_history.py
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
import redis
from redis.exceptions import ConnectionError, TimeoutError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.core.exceptions import ClientConnectionError
from src.clients.base import BaseChatHistory

logger = logging.getLogger(__name__)

class RedisLangChainHistory(BaseChatHistory):
    def __init__(self, config):
        self.config = config.redis
        self.redis_config = {
            'host': self.config.host,
            'port': self.config.port,
            'db': self.config.db,
            'decode_responses': True,
            'socket_timeout': 5,
            'retry_on_timeout': True
        }
        self.max_retries = self.config.max_retries
        self.retry_interval = self.config.retry_interval
        self.redis_client = self._connect_with_retry()

    def _connect_with_retry(self) -> redis.Redis:
        """Establish Redis connection with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                client = redis.Redis(**self.redis_config)
                client.ping()
                logger.info("Successfully connected to Redis")
                return client
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.retry_interval)
                else:
                    raise ClientConnectionError("Failed to connect to Redis after maximum retries")

    def _message_to_dict(self, message: BaseMessage) -> Dict:
        """Convert LangChain message to dictionary format."""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }

    def _dict_to_message(self, data: Dict) -> BaseMessage:
        """Convert dictionary to LangChain message."""
        content = data["content"]
        if data["type"] == "HumanMessage":
            return HumanMessage(content=content)
        elif data["type"] == "AIMessage":
            return AIMessage(content=content)
        else:
            raise ValueError(f"Unknown message type: {data['type']}")

    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = f"chat:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.redis_client.hset(
            f"{session_id}:meta",
            mapping={
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        )
        return session_id

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Add a LangChain message to the session."""
        message_dict = self._message_to_dict(message)
        self.redis_client.rpush(session_id, json.dumps(message_dict))
        self.redis_client.hincrby(f"{session_id}:meta", "message_count", 1)

    def get_session_messages(self, session_id: str) -> List[BaseMessage]:
        """Get all messages from a session as LangChain messages."""
        messages = self.redis_client.lrange(session_id, 0, -1)
        return [self._dict_to_message(json.loads(msg)) for msg in messages]

    def get_session_info(self, session_id: str) -> Dict:
        """Get session metadata."""
        meta = self.redis_client.hgetall(f"{session_id}:meta")
        message_count = self.redis_client.llen(session_id)
        return {
            "created_at": meta.get("created_at", "Unknown"),
            "message_count": message_count
        }

    def list_sessions(self) -> List[Dict]:
        """List all available chat sessions with metadata."""
        sessions = []
        for key in self.redis_client.keys("chat:*"):
            if ":meta" not in key:
                session_info = self.get_session_info(key)
                sessions.append({
                    "id": key,
                    "created_at": session_info.get("created_at"),
                    "message_count": session_info.get("message_count", 0)
                })
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its metadata."""
        try:
            self.redis_client.delete(session_id, f"{session_id}:meta")
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def clear_all_sessions(self) -> bool:
        """Clear all chat sessions."""
        try:
            for key in self.redis_client.keys("chat:*"):
                self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error clearing sessions: {e}")
            return False