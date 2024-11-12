# src/utils/redis_history.py
import redis
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisChatHistory:
    def __init__(self, host='localhost', port=6379, db=0, max_retries=3, retry_interval=1):
        self.redis_config = {
            'host': host,
            'port': port,
            'db': db,
            'decode_responses': True,
            'socket_timeout': 5,
            'retry_on_timeout': True
        }
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.redis_client = self._connect_with_retry()
        self.current_session_id = None

    def _connect_with_retry(self) -> redis.Redis:
        """Establish Redis connection with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                client = redis.Redis(**self.redis_config)
                client.ping()  # Test connection
                logger.info("Successfully connected to Redis")
                return client
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_interval)
                else:
                    logger.error("Failed to connect to Redis after maximum retries")
                    raise

    def _ensure_connection(self):
        """Ensure Redis connection is active."""
        try:
            self.redis_client.ping()
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection lost, attempting to reconnect...")
            self.redis_client = self._connect_with_retry()

    def create_session(self) -> str:
        """Create a new chat session."""
        self._ensure_connection()
        session_id = f"chat:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        # Store session metadata
        self.redis_client.hset(
            f"{session_id}:meta",
            mapping={
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        )
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to a specific chat session."""
        self._ensure_connection()
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            # Push message to the session list
            self.redis_client.rpush(session_id, json.dumps(message))
            # Update message count in metadata
            self.redis_client.hincrby(f"{session_id}:meta", "message_count", 1)
        except Exception as e:
            logger.error(f"Error adding message to Redis: {e}")

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages from a specific session."""
        self._ensure_connection()
        try:
            messages = self.redis_client.lrange(session_id, 0, -1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            logger.error(f"Error getting messages from Redis: {e}")
            return []

    def get_session_info(self, session_id: str) -> Dict:
        """Get session metadata."""
        self._ensure_connection()
        try:
            meta = self.redis_client.hgetall(f"{session_id}:meta")
            message_count = self.redis_client.llen(session_id)
            return {
                "created_at": meta.get("created_at", "Unknown"),
                "message_count": message_count
            }
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {}

    def list_sessions(self) -> List[Dict]:
        """List all available chat sessions with metadata."""
        self._ensure_connection()
        try:
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
        except Exception as e:
            logger.error(f"Error listing sessions from Redis: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a specific chat session and its metadata."""
        self._ensure_connection()
        try:
            # Delete both session messages and metadata
            self.redis_client.delete(session_id, f"{session_id}:meta")
            return True
        except Exception as e:
            logger.error(f"Error deleting session from Redis: {e}")
            return False

    def clear_all_sessions(self) -> bool:
        """Clear all chat sessions."""
        self._ensure_connection()
        try:
            for key in self.redis_client.keys("chat:*"):
                self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error clearing all sessions: {e}")
            return False