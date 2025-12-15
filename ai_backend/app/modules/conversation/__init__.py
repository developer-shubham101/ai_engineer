"""Conversation history management module."""

from .conversation_manager import IConversationManager, SQLiteConversationManager

__all__ = ["IConversationManager", "SQLiteConversationManager"]
