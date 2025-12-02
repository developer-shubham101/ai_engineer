# app/api_routes_auth.py
"""
Authentication API routes.
Handles user login and token generation.
"""
import logging
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.modules.auth.interfaces import ISessionManager
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Request/Response Models
class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


# Endpoints
@router.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    """
    Login endpoint - authenticate user and generate JWT token.
    
    Request:
        {
            "username": "admin",
            "password": "admin123"
        }
    
    Response:
        {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
            "token_type": "bearer",
            "user": {
                "user_id": "u_admin_1",
                "username": "admin",
                "role": "SuperAdmin",
                "department": "Executive"
            }
        }
    """
    container = get_container()
    container.initialize()

    # Authenticate user
    authenticator = container.get_authenticator()
    user_data = await authenticator.authenticate(request.username, request.password)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate a new session ID for this login
    session_id = f"sess_{uuid.uuid4().hex}"

    # Generate access token with session_id
    access_token = await authenticator.create_access_token(user_data, session_id=session_id)

    # Auto-create support chat session using the new session_id
    try:
        session_manager: ISessionManager = container.get_session_manager()

        session_manager.create_session(
            session_id=session_id,
            role=user_data["role"],
            department=user_data["department"]
        )
        logger.info(f"Created support session {session_id} for user {user_data['user_id']}")
    except Exception as e:
        # Session might already exist, that's okay
        logger.debug(f"Session creation skipped or failed: {e}")

    # Get user profile from user_meta
    user_manager = container.get_user_manager()
    profile = user_data.get("profile", {})

    # Add profile to user data
    user_response = {**user_data, "profile": profile}

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )
