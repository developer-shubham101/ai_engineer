# app/api_routes_conversations.py
"""
Conversation history API routes.
Supports filtering by history type: rag | agent | crew
"""

import logging
from typing import List, Optional, Dict, Any, Literal

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.modules.integration import get_container
from app.logging_config import log_user_action

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

VALID_HISTORY_TYPES = {"rag", "agent", "crew"}


# ── Models ────────────────────────────────────────────────────────────────────

class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: Optional[int] = 0


class RagMessageResponse(BaseModel):
    """RAG pipeline message with full retrieval + LLM logging."""
    id: int
    speaker: str
    content: str
    created_at: str
    sentiment: Optional[str] = None
    tone: Optional[str] = None
    sentiment_meta: Optional[Dict[str, Any]] = None
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


class AgentMessageResponse(BaseModel):
    """Agent workflow message with steps + tools logging."""
    id: int
    speaker: str
    content: str
    created_at: str
    user_query: Optional[str] = None
    tools_used: Optional[List[str]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    orchestrator_type: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class CrewMessageResponse(BaseModel):
    """CrewAI workflow message with agents + workflow logging."""
    id: int
    speaker: str
    content: str
    created_at: str
    user_topic: Optional[str] = None
    workflow_type: Optional[str] = None
    agents_used: Optional[List[str]] = None
    iterations: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_manager():
    container = get_container()
    container.initialize()
    return container.get_conversation_manager()


# ── Conversation CRUD ─────────────────────────────────────────────────────────

@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all conversations for the authenticated user."""
    try:
        conversations = await _get_manager().list_conversations(
            user_id=current_user["user_id"], limit=limit, offset=offset
        )
        logger.debug("CONV_LIST: user=%s | count=%d", current_user["user_id"], len(conversations))
        return [ConversationResponse(**c) for c in conversations]
    except Exception as e:
        logger.exception("CONV_LIST: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    title: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new conversation."""
    try:
        mgr = _get_manager()
        conv_id = await mgr.create_conversation(user_id=current_user["user_id"], title=title)
        conversation = await mgr.get_conversation(conv_id, current_user["user_id"])
        log_user_action(logger, "CONV_CREATED", current_user["user_id"], conversation_id=conv_id)
        return ConversationResponse(**conversation)
    except Exception as e:
        logger.exception("CONV_CREATE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific conversation."""
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
    """Update conversation metadata (e.g., rename)."""
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
    """Delete a conversation (soft delete)."""
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


# ── Messages with history_type filter ────────────────────────────────────────

@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    history_type: str = Query(
        default="rag",
        description="Filter message history by source: `rag` (default), `agent`, or `crew`"
    ),
    limit: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get messages from a conversation filtered by history type.

    - **rag** (default): RAG pipeline messages — retrieved docs, LLM prompt, provider, tokens
    - **agent**: Agent workflow messages — tools used, step-by-step execution log
    - **crew**: CrewAI workflow messages — agents used, workflow type, iterations
    """
    if history_type not in VALID_HISTORY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid history_type '{history_type}'. Valid values: {sorted(VALID_HISTORY_TYPES)}"
        )

    user_id = current_user["user_id"]
    logger.debug("CONV_MESSAGES: conversation_id=%s | history_type=%s | user=%s", conversation_id, history_type, user_id)

    try:
        mgr = _get_manager()

        if history_type == "rag":
            messages = await mgr.get_messages(conversation_id, user_id, limit=limit)
            logger.info("CONV_MESSAGES: rag | count=%d | conversation_id=%s", len(messages), conversation_id)
            return {
                "conversation_id": conversation_id,
                "history_type": "rag",
                "messages": [RagMessageResponse(**m) for m in messages],
                "count": len(messages)
            }

        elif history_type == "agent":
            messages = await mgr.get_agent_messages(conversation_id, user_id, limit=limit)
            logger.info("CONV_MESSAGES: agent | count=%d | conversation_id=%s", len(messages), conversation_id)
            return {
                "conversation_id": conversation_id,
                "history_type": "agent",
                "messages": [AgentMessageResponse(**m) for m in messages],
                "count": len(messages)
            }

        elif history_type == "crew":
            messages = await mgr.get_crew_messages(conversation_id, user_id, limit=limit)
            logger.info("CONV_MESSAGES: crew | count=%d | conversation_id=%s", len(messages), conversation_id)
            return {
                "conversation_id": conversation_id,
                "history_type": "crew",
                "messages": [CrewMessageResponse(**m) for m in messages],
                "count": len(messages)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_MESSAGES: failed | history_type=%s | error=%s", history_type, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Restore a conversation to the current session."""
    try:
        mgr = _get_manager()
        conversation = await mgr.get_conversation(conversation_id, current_user["user_id"])
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        session_id = current_user.get("session_id")
        logger.info("CONV_RESTORE: conversation_id=%s | user=%s | session_id=%s",
                    conversation_id, current_user["user_id"], session_id)

        return {"message": "Conversation restored successfully", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CONV_RESTORE: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))
