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
    async def create_access_token(self, user_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Create authentication token."""
        pass

    @abstractmethod
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token."""
        pass


class IUserManager(ABC):
    """Interface for user management operations."""
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        pass
    
    @abstractmethod
    async def create_user(self, username: str, password: str, role: str, department: Optional[str] = None, profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user."""
        pass

    @abstractmethod
    async def update_user(self, user_id: str, **kwargs: Any) -> bool:
        """Update user fields."""
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
    """Interface for managing user sessions, messages, and profiles (including support chat logic)."""

    # Session Management
    @abstractmethod
    def create_session(self, session_id: Optional[str], role: Optional[str], department: Optional[str]) -> str:
        """Create a new support chat session."""
        pass

    @abstractmethod
    async def touch_session(self, session_id: str) -> None:
        """Update session metadata/timestamp."""
        pass

    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """End and delete a support chat session and its data."""
        pass

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check if a session ID exists."""
        pass

    # Message Storage
    @abstractmethod
    def store_message(self, session_id: str, speaker: str, content: str) -> int:
        """Store a message and compute sentiment/tone."""
        pass

    @abstractmethod
    def fetch_recent_messages(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent messages for a session."""
        pass

    @abstractmethod
    def render_history(self, messages: List[Dict[str, Any]]) -> str:
        """Render message history into a single string."""
        pass

    # Profile Management
    @abstractmethod
    def set_profile_value(self, session_id: str, key: str, value: str) -> None:
        """Set a single profile key/value for the session."""
        pass

    @abstractmethod
    def get_profile_value(self, session_id: str, key: str) -> Optional[str]:
        """Get a single profile value for the session."""
        pass

    @abstractmethod
    def get_full_profile(self, session_id: str) -> Dict[str, str]:
        """Get the full profile dictionary for the session."""
        pass

    @abstractmethod
    def load_onboarding_fields(self) -> List[Dict[str, str]]:
        """Load configured onboarding fields."""
        pass

    @abstractmethod
    def get_next_missing_profile_key(self, session_id: str) -> Optional[Dict[str, str]]:
        """Get the next missing profile key for onboarding."""
        pass

    # Analytics
    @abstractmethod
    def get_sentiment_stats(self) -> Dict[str, Dict]:
        """Get aggregated sentiment and tone statistics."""
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
