"""Authentication and session management interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime


class IAuthenticator(ABC):
    """Interface for authentication providers."""
    
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user data."""
        pass
    
    @abstractmethod
    async def create_token(self, user_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Create authentication token."""
        pass
    
    @abstractmethod
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return user data."""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke authentication token."""
        pass


class IUserManager(ABC):
    """Interface for user management."""
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        pass
    
    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user and return user ID."""
        pass
    
    @abstractmethod
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """Update user data."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    async def get_user_metadata(self, user_id: str, key: str) -> Any:
        """Get user metadata value."""
        pass
    
    @abstractmethod
    async def set_user_metadata(self, user_id: str, key: str, value: Any) -> bool:
        """Set user metadata value."""
        pass


class ISessionManager(ABC):
    """Interface for session management."""
    
    @abstractmethod
    async def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new session and return session ID."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        pass
    
    @abstractmethod
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        pass
    
    @abstractmethod
    async def store_message(self, session_id: str, speaker: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store message in session history."""
        pass
    
    @abstractmethod
    async def get_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session message history."""
        pass
    
    @abstractmethod
    async def clear_messages(self, session_id: str) -> bool:
        """Clear session message history."""
        pass


class IRBACManager(ABC):
    """Interface for Role-Based Access Control."""
    
    @abstractmethod
    async def check_permission(self, user: Dict[str, Any], resource: str, action: str) -> bool:
        """Check if user has permission for action on resource."""
        pass
    
    @abstractmethod
    async def filter_documents(self, documents: List[Dict[str, Any]], user: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter documents based on user permissions."""
        pass
    
    @abstractmethod
    async def can_access_document(self, document_metadata: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Check if user can access specific document."""
        pass
    
    @abstractmethod
    async def get_user_level(self, user: Dict[str, Any]) -> int:
        """Get user's access level."""
        pass
    
    @abstractmethod
    async def log_access_attempt(self, user: Dict[str, Any], resource: str, action: str, granted: bool) -> None:
        """Log access attempt for audit."""
        pass