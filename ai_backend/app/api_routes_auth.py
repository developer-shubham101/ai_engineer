# app/api_routes_auth.py
"""
Authentication API routes.
Handles user login and token generation.
"""
from typing import Dict, Any
import logging
import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.legacy.user_service import authenticate_user, get_all_user_meta
from app.services.legacy.auth import create_access_token
from app.services.legacy import support_chat

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
    # Authenticate user
    user_data = authenticate_user(request.username, request.password)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate a new session ID for this login
    session_id = f"sess_{uuid.uuid4().hex}"

    # Generate access token with session_id
    access_token = create_access_token(user_data, session_id=session_id)
    
    # Auto-create support chat session using the new session_id
    try:
        support_chat.create_session(
            session_id=session_id,
            role=user_data["role"],
            department=user_data["department"]
        )
        logger.info(f"Created support session {session_id} for user {user_data['user_id']}")
    except Exception as e:
        # Session might already exist (unlikely with UUID), that's okay
        logger.debug(f"Session creation skipped or failed: {e}")
    
    # Get user profile from user_meta
    profile = get_all_user_meta(user_data["user_id"])
    
    # Add profile to user data
    user_response = {**user_data, "profile": profile}
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )
