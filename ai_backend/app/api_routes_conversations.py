# app/api_routes_conversations.py
"""
Conversation history API routes.
- chat_type (rag | agent | crew) is REQUIRED when creating a conversation
- list supports optional ?chat_type= filter
- GET /{id}/messages returns messages from the single messages table (filtered by conversation)
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.modules.integration import get_container
from app.modules.conversation.conversation_manager import VALID_CHAT_TYPES
from app.logging_config import log_user_action

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


# ── Models ────────────────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    chat_type: str                  # required: rag | agent | crew
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    chat_type: str
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: Optional[int] = 0


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None


class MessageResponse(BaseModel):
    """Unified message response — all chat_type fields present, unused ones are null."""
    id: int
    conversation_id: str
    chat_type: str
    speaker: str
    content: str
    created_at: str
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    # RAG fields
    user_query: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt: Optional[str] = None
    llm_response_raw: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    use_documents: Optional[bool] = None
    use_llm: Optional[bool] = None
    sentiment: Optional[str] = None
    tone: Optional[str] = None
    retrieved_context: Optional[List[Dict[str, Any]]] = None
    embeddings_used: Optional[Dict[str, Any]] = None
    retrieved_doc_ids: Optional[List[str]] = None
    sentiment_meta: Optional[Dict[str, Any]] = None
    # Agent fields
    orchestrator_type: Optional[str] = None
    tools_used: Optional[List[str]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    # Crew fields
    workflow_type: Optional[str] = None
    iterations: Optional[int] = None
    agents_used: Optional[List[str]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_manager():
    container = get_container()
    container.initialize()
    return container.get_conversation_manager()


def _validate_chat_type(chat_type: str):
    if chat_type not in VALID_CHAT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chat_type '{chat_type}'. Valid values: {sorted(VALID_CHAT_TYPES)}"
        )


# ── Conversation CRUD ─────────────────────────────────────────────────────────

@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    chat_type: Optional[str] = Query(default=None, description="Filter by chat_type: rag | agent | crew"),
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List conversations. Optionally filter by chat_type."""
    if chat_type:
        _validate_chat_type(chat_type)

    try:
        conversations = await _get_manager().list_conversations(
            user_id=current_user["user_id"],
            chat_type=chat_type,
            limit=limit,
            offset=offset
        )
        logger.debug("CONV_LIST: user=%s | chat_type=%s | count=%d",
                     current_user["user_id"], chat_type or "all", len(conversations))
        return [ConversationResponse(**c) for c in conversations]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_LIST: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new conversation. chat_type (rag | agent | crew) is required."""
    _validate_chat_type(request.chat_type)

    try:
        mgr = _get_manager()
        conv_id = await mgr.create_conversation(
            user_id=current_user["user_id"],
            chat_type=request.chat_type,
            title=request.title
        )
        conversation = await mgr.get_conversation(conv_id, current_user["user_id"])
        log_user_action(logger, "CONV_CREATED", current_user["user_id"],
                        conversation_id=conv_id, chat_type=request.chat_type)
        return ConversationResponse(**conversation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("CONV_CREATE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific conversation (includes chat_type)."""
    try:
        conversation = await _get_manager().get_conversation(conversation_id, current_user["user_id"])
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return ConversationResponse(**conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_GET: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Rename a conversation."""
    try:
        mgr = _get_manager()
        success = await mgr.update_conversation(conversation_id, current_user["user_id"], title=request.title)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation = await mgr.get_conversation(conversation_id, current_user["user_id"])
        return ConversationResponse(**conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_UPDATE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Soft-delete a conversation."""
    try:
        success = await _get_manager().delete_conversation(conversation_id, current_user["user_id"])
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        log_user_action(logger, "CONV_DELETED", current_user["user_id"], conversation_id=conversation_id)
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_DELETE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Messages ──────────────────────────────────────────────────────────────────

@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all messages for a conversation from the unified messages table.

    Messages are already scoped to the conversation's chat_type — no extra filter needed.
    Each message includes a `chat_type` field so the client knows which fields are populated:
    - **rag**: llm_provider, llm_model, retrieved_context, llm_prompt, ...
    - **agent**: orchestrator_type, tools_used, steps
    - **crew**: workflow_type, agents_used, iterations
    """
    try:
        mgr = _get_manager()

        # Verify conversation exists and belongs to user
        conv = await mgr.get_conversation(conversation_id, current_user["user_id"])
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = await mgr.get_messages(conversation_id, current_user["user_id"], limit=limit)

        logger.info("CONV_MESSAGES: conversation_id=%s | chat_type=%s | count=%d | user=%s",
                    conversation_id, conv.get("chat_type"), len(messages), current_user["user_id"])

        return {
            "conversation_id": conversation_id,
            "chat_type": conv.get("chat_type"),
            "messages": [MessageResponse(**m) for m in messages],
            "count": len(messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_MESSAGES: failed | conversation_id=%s | error=%s", conversation_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Restore a conversation to the current session."""
    try:
        conv = await _get_manager().get_conversation(conversation_id, current_user["user_id"])
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        logger.info("CONV_RESTORE: conversation_id=%s | user=%s", conversation_id, current_user["user_id"])
        return {"message": "Conversation restored successfully", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_RESTORE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))
