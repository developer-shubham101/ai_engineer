"""API request handlers and response formatters."""

from typing import Dict, Any, Optional
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


class APIHandler:
    """Base API handler with common functionality."""
    
    @staticmethod
    def format_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Format successful API response."""
        return {
            "success": True,
            "message": message,
            "data": data
        }
    
    @staticmethod
    def format_error_response(error: str, detail: Optional[str] = None, code: Optional[str] = None) -> Dict[str, Any]:
        """Format error API response."""
        return {
            "success": False,
            "error": error,
            "detail": detail,
            "code": code
        }
    
    @staticmethod
    def handle_exception(e: Exception, context: str = "") -> HTTPException:
        """Convert exceptions to HTTP exceptions."""
        logger.error(f"API error in {context}: {str(e)}")
        
        if isinstance(e, HTTPException):
            return e
        elif isinstance(e, ValueError):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        elif isinstance(e, PermissionError):
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )
        else:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )


class RAGHandler(APIHandler):
    """Handler for RAG-specific API operations."""
    
    @staticmethod
    def validate_query_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize query request."""
        # Ensure required fields
        if not request.get("question"):
            raise ValueError("Question is required")
        
        # Normalize parameters
        request["top_k"] = max(1, min(request.get("top_k", 3), 10))
        request["max_tokens"] = max(1, min(request.get("max_tokens", 256), 2048))
        request["temperature"] = max(0.0, min(request.get("temperature", 0.1), 1.0))
        
        return request


class AuthHandler(APIHandler):
    """Handler for authentication API operations."""
    
    @staticmethod
    def validate_auth_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate authentication request."""
        if not request.get("username"):
            raise ValueError("Username is required")
        if not request.get("password"):
            raise ValueError("Password is required")
        
        return request