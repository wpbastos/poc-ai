# src/utils/chat_history.py
import json
import os
from datetime import datetime
from typing import List, Dict

class ChatHistory:
    def __init__(self, history_path: str):
        self.history_path = history_path
        self.current_session: List[Dict] = []
        self._ensure_history_directory()
        
    def _ensure_history_directory(self):
        """Create history directory if it doesn't exist."""
        os.makedirs(self.history_path, exist_ok=True)
        
    def add_message(self, role: str, content: str):
        """Add a message to the current session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.current_session.append(message)
        
    def save_session(self):
        """Save current session to a file."""
        if not self.current_session:
            return
            
        filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.history_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2)
            
    def load_session(self, filename: str) -> List[Dict]:
        """Load a specific session from file."""
        filepath = os.path.join(self.history_path, filename)
        
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def get_current_session(self) -> List[Dict]:
        """Get messages from current session."""
        return self.current_session