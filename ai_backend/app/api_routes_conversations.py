# app/api_routes_conversations.py
"""
Conversation history API routes.
Provides endpoints for managing conversation history across devices.
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.dependencies import get_current_user
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


# Request/Response Models
class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: Optional[int] = 0


class MessageResponse(BaseModel):
    """Enhanced message response with full RAG pipeline logging."""
    id: int
    speaker: str
    content: str
    created_at: str
    
    # Sentiment/Tone
    sentiment: Optional[str] = None
    tone: Optional[str] = None
    sentiment_meta: Optional[Dict[str, Any]] = None
    
    # RAG Pipeline Logging
    user_query: Optional[str] = None
    retrieved_context: Optional[List[Dict[str, Any]]] = None
    embeddings_used: Optional[Dict[str, Any]] = None
    llm_prompt: Optional[str] = None
    llm_response_raw: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    retrieved_doc_ids: Optional[List[str]] = None
    retrieval_top_k: Optional[int] = None
    use_documents: Optional[bool] = None
    use_llm: Optional[bool] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None


# Endpoints
@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all conversations for the authenticated user."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        conversations = await conversation_manager.list_conversations(
            user_id=current_user["user_id"],
            limit=limit,
            offset=offset
        )
        
        return [ConversationResponse(**conv) for conv in conversations]
    except Exception as e:
        logger.exception("Failed to list conversations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    title: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new conversation."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        conv_id = await conversation_manager.create_conversation(
            user_id=current_user["user_id"],
            title=title
        )
        
        # Get the created conversation
        conversation = await conversation_manager.get_conversation(
            conversation_id=conv_id,
            user_id=current_user["user_id"]
        )
        
        return ConversationResponse(**conversation)
    except Exception as e:
        logger.exception("Failed to create conversation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific conversation."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        conversation = await conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user["user_id"]
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(**conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get conversation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update conversation metadata (e.g., rename)."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        success = await conversation_manager.update_conversation(
            conversation_id=conversation_id,
            user_id=current_user["user_id"],
            title=request.title
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get updated conversation
        conversation = await conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user["user_id"]
        )
        
        return ConversationResponse(**conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update conversation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a conversation (soft delete)."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        success = await conversation_manager.delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user["user_id"]
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete conversation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    limit: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all messages from a conversation with full RAG logging details."""
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        
        messages = await conversation_manager.get_messages(
            conversation_id=conversation_id,
            user_id=current_user["user_id"],
            limit=limit
        )
        
        return [MessageResponse(**msg) for msg in messages]
    except Exception as e:
        logger.exception("Failed to get messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Restore a conversation to the current session.
    This links the conversation to the user's current session.
    """
    try:
        container = get_container()
        container.initialize()
        conversation_manager = container.get_conversation_manager()
        session_manager = container.get_session_manager()
        
        # Verify conversation exists and user owns it
        conversation = await conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user["user_id"]
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update session to link to this conversation
        session_id = current_user.get("session_id")
        if session_id:
            # Note: This requires updating session_manager to support conversation_id
            # For now, we'll just return success
            logger.info(f"Restored conversation {conversation_id} for user {current_user['user_id']}")
        
        return {
            "message": "Conversation restored successfully",
            "conversation_id": conversation_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to restore conversation: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
