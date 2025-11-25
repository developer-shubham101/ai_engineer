# app/api_routes_auth.py
"""
Authentication API routes.
Handles user login and token generation.
"""
from typing import Dict, Any
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.user_service import authenticate_user, get_all_user_meta
from app.services.auth import create_access_token
from app.services import support_chat

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
    
    # Generate access token
    access_token = create_access_token(user_data)
    
    # Auto-create support chat session using user_id
    try:
        support_chat.create_session(
            session_id=user_data["user_id"],  # Use user_id as session_id
            role=user_data["role"],
            department=user_data["department"]
        )
        logger.info(f"Created support session for user {user_data['user_id']}")
    except Exception as e:
        # Session might already exist, that's okay
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
