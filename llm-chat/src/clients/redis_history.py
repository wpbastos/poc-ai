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
            "content": message.content if message.content else "",  # Ensure content is not None
            "timestamp": datetime.now().isoformat()
        }

    def create_session(self, chat_name: str = None) -> str:
        """Create a new chat session."""
        session_id = f"chat:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure chat_name is not None
        if chat_name is None:
            chat_name = f"Chat {datetime.now().strftime('%Y%m%d_%H%M')}"
            
        metadata = {
            "created_at": datetime.now().isoformat(),
            "message_count": "0",  # Convert to string for Redis
            "chat_name": chat_name
        }
        
        # Set metadata
        try:
            self.redis_client.hset(f"{session_id}:meta", mapping=metadata)
            logger.info(f"Created new session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Add a message to the session."""
        if not session_id or not message:
            logger.error("Invalid session_id or message")
            return
            
        try:
            message_dict = self._message_to_dict(message)
            message_json = json.dumps(message_dict)
            self.redis_client.rpush(session_id, message_json)
            
            # Update message count
            count_key = f"{session_id}:meta"
            current_count = self.redis_client.hget(count_key, "message_count")
            new_count = int(current_count if current_count else 0) + 1
            self.redis_client.hset(count_key, "message_count", str(new_count))
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise

    def get_session_messages(self, session_id: str) -> List[BaseMessage]:
        """Get all messages from a session."""
        if not session_id:
            logger.error("Invalid session_id")
            return []
            
        try:
            messages = self.redis_client.lrange(session_id, 0, -1)
            result = []
            
            for msg in messages:
                if msg:
                    try:
                        msg_dict = json.loads(msg)
                        if msg_dict["type"] == "HumanMessage":
                            result.append(HumanMessage(content=msg_dict["content"]))
                        elif msg_dict["type"] == "AIMessage":
                            result.append(AIMessage(content=msg_dict["content"]))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode message: {msg}")
                        continue
                        
            return result
            
        except Exception as e:
            logger.error(f"Error getting session messages: {str(e)}")
            return []

    def get_session_info(self, session_id: str) -> Dict:
        """Get session metadata."""
        if not session_id:
            return {"created_at": datetime.now().isoformat(), "message_count": "0", "chat_name": "New Chat"}
            
        try:
            meta = self.redis_client.hgetall(f"{session_id}:meta") or {}
            message_count = self.redis_client.llen(session_id) or 0
            
            return {
                "created_at": meta.get("created_at", datetime.now().isoformat()),
                "message_count": str(message_count),
                "chat_name": meta.get("chat_name", f"Chat {session_id.split(':')[1]}")
            }
            
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}")
            return {"created_at": datetime.now().isoformat(), "message_count": "0", "chat_name": "New Chat"}

    def update_chat_name(self, session_id: str, chat_name: str) -> bool:
        """Update the name of an existing chat session."""
        if not session_id or not chat_name:
            logger.error("Invalid session_id or chat_name")
            return False
            
        try:
            self.redis_client.hset(f"{session_id}:meta", "chat_name", chat_name)
            return True
        except Exception as e:
            logger.error(f"Error updating chat name: {str(e)}")
            return False

    def list_sessions(self) -> List[Dict]:
        """List all available chat sessions with metadata."""
        try:
            sessions = []
            for key in self.redis_client.keys("chat:*"):
                if ":meta" not in key:
                    session_info = self.get_session_info(key)
                    sessions.append({
                        "id": key,
                        "created_at": session_info.get("created_at"),
                        "message_count": session_info.get("message_count", "0"),
                        "chat_name": session_info.get("chat_name", f"Chat {key.split(':')[1]}")
                    })
            return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its metadata."""
        if not session_id:
            return False
            
        try:
            self.redis_client.delete(session_id, f"{session_id}:meta")
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False

    def clear_all_sessions(self) -> bool:
        """Clear all chat sessions."""
        try:
            for key in self.redis_client.keys("chat:*"):
                self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error clearing sessions: {str(e)}")
            return False