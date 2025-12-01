"""JWT-based authentication implementation."""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from .interfaces import IAuthenticator
from ..config.settings import settings

logger = logging.getLogger(__name__)


class JWTAuthenticator(IAuthenticator):
    """JWT-based authentication implementation."""
    
    def __init__(self, user_manager=None):
        self.user_manager = user_manager
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expiration_days = settings.JWT_EXPIRATION_DAYS
    
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password."""
        if not self.user_manager:
            raise RuntimeError("User manager not configured")
        
        user = await self.user_manager.get_user_by_username(username)
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        # Simple password check (in production, use proper hashing)
        if user.get("password") != password:
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None
        
        logger.info(f"User authenticated successfully: {username}")
        return user
    
    async def create_token(self, user_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "department": user_data.get("department"),
            "exp": datetime.utcnow() + timedelta(days=self.expiration_days),
            "iat": datetime.utcnow()
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token created for user: {user_data['username']}")
        return token
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke token (placeholder - implement token blacklist if needed)."""
        # In a production system, you'd maintain a blacklist of revoked tokens
        logger.info("Token revocation requested")
        return True