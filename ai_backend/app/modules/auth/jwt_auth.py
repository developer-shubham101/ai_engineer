"""JWT-based authentication implementation."""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from .interfaces import IAuthenticator
from ..config.settings import settings
from app.logging_config import log_user_action, log_sensitive_debug, log_security_event

logger = logging.getLogger(__name__)


class JWTAuthenticator(IAuthenticator):
    """
    JWT-based implementation of IAuthenticator.
    Relies on IUserManager for user authentication lookups.
    """

    def __init__(self):
        # user_manager is not available at construction, it's injected by the container
        self.user_manager = None
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expiration_days = settings.JWT_EXPIRATION_DAYS

    # ---------------------------------------------------------
    # IAuthenticator Implementation
    # ---------------------------------------------------------

    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user against the UserManager.
        """
        if not self.user_manager:
            raise RuntimeError("UserManager not initialized in JWTAuthenticator")

        user = await self.user_manager.authenticate(username, password)
        
        if user:
            # Remove password hash for safety before returning
            user.pop('password', None) # user_manager.authenticate already removes it, but as a safeguard.
            return user

        return None


    async def create_access_token(self, user_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Create a JWT access token for a user.
        """
        # Calculate expiration time
        expire = datetime.utcnow() + timedelta(days=self.expiration_days)

        # Create token payload
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data.get("username"),
            "role": user_data["role"],
            "department": user_data["department"],
            "session_id": session_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }

        # Encode token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Logging
        username = user_data.get('username')
        user_id = user_data.get('user_id')
        role = user_data.get('role')

        log_user_action(
            logger, "TOKEN_CREATED", user_id,
            username=username, role=role, expires_days=self.expiration_days,
            session_id=session_id
        )

        log_sensitive_debug(
            logger, "JWT token created",
            token_preview=token[:20] + "...", payload_keys=list(payload.keys()),
            user_data=user_data
        )

        return token


    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            log_security_event(logger, "TOKEN_EXPIRED", token_preview=token[:20] + "...")
            return None
        except jwt.InvalidTokenError as e:
            log_security_event(logger, "INVALID_TOKEN", error=str(e), token_preview=token[:20] + "...")
            return None
