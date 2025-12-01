# app/services/auth.py
"""
Authentication service with JWT token support.
Handles token generation, verification, and user authentication.
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_DAYS

logger = logging.getLogger(__name__)

def create_access_token(user_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        user_data: Dict containing user_id, username, role, department
        session_id: Optional session ID to include in token
        
    Returns:
        JWT token string
    """
    # Calculate expiration time
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS)
    
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
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    from app.logging_config import log_user_action, log_sensitive_debug
    
    username = user_data.get('username')
    user_id = user_data.get('user_id')
    role = user_data.get('role')
    
    log_user_action(
        logger, "TOKEN_CREATED", user_id,
        username=username, role=role, expires_days=JWT_EXPIRATION_DAYS,
        session_id=session_id
    )
    
    # Log sensitive token info for debugging (remove in production)
    log_sensitive_debug(
        logger, "JWT token created",
        token_preview=token[:20] + "...", payload_keys=list(payload.keys()),
        user_data=user_data
    )
    
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid
        None if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        from app.logging_config import log_security_event
        log_security_event(logger, "TOKEN_EXPIRED", token_preview=token[:20] + "...")
        return None
    except jwt.InvalidTokenError as e:
        from app.logging_config import log_security_event
        log_security_event(logger, "INVALID_TOKEN", error=str(e), token_preview=token[:20] + "...")
        return None
