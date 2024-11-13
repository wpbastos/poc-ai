# src/utils/formatting.py
from datetime import datetime
from typing import Dict, Any

def format_session_info(session: Dict[str, Any]) -> str:
    """Format session information for display."""
    created_at = datetime.fromisoformat(session["created_at"])
    chat_name = session.get("chat_name", f"Chat {session['id'].split(':')[1]}")
    return f"{chat_name} - {created_at.strftime('%Y-%m-%d %H:%M')} ({session['message_count']} messages)"

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")