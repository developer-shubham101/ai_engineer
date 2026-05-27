"""Agent API routes."""
from __future__ import annotations

import inspect
import json as _json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_current_user_optional
from app.modules.agents.interfaces import AgentRequest
from app.modules.agents.orchestrators.utils import get_tool_registry
from app.modules.integration import get_container
from app.logging_config import log_user_action, log_performance_metric, log_security_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentQueryRequest(BaseModel):
    question: str
    workflow: str = "smart_assistant"
    tools: List[str] = Field(default_factory=list)
    max_steps: int = 5
    temperature: float = 0.1
    provider: str = "local"
    orchestrator_type: str = "autogen"   # autogen | custom | mcp | crewai
    conversation_id: Optional[str] = None
    debug: bool = False


class AgentQueryResponse(BaseModel):
    answer: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    available_workflows: List[str] = Field(default_factory=list)
    orchestrator_type: str = ""
    conversation_id: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolTestRequest(BaseModel):
    input_data: str


class AgentStatusResponse(BaseModel):
    orchestrator_types: Dict[str, bool]
    tools: List[ToolInfo]
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_info_list() -> List[ToolInfo]:
    """Build ToolInfo list from the shared utils tool registry."""
    return [
        ToolInfo(name=name, description=inspect.getdoc(fn) or name)
        for name, fn in get_tool_registry().items()
    ]


def _create_orchestrator(orchestrator_type: str):
    from app.modules.agents.factories import AgentOrchestratorFactory
    return AgentOrchestratorFactory.create_orchestrator(orchestrator_type=orchestrator_type)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get agent system status, available orchestrators and tools."""
    try:
        from app.modules.agents.factories import AgentOrchestratorFactory
        return AgentStatusResponse(
            orchestrator_types=AgentOrchestratorFactory.get_available_types(),
            tools=_tool_info_list(),
            status="active",
        )
    except Exception as e:
        logger.error("AGENT_STATUS: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows(orchestrator_type: str = "autogen"):
    """List available workflows and tools for a given orchestrator type.

    **orchestrator_type**: `autogen` | `custom` | `mcp` | `crewai`
    """
    try:
        orch = _create_orchestrator(orchestrator_type)
        return {
            "orchestrator_type": orchestrator_type,
            "workflows": orch.get_available_workflows(),
            "tools": orch.get_available_tools(),
        }
    except Exception as e:
        logger.error("AGENT_WORKFLOWS: failed | orchestrator=%s | error=%s", orchestrator_type, e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tools", response_model=List[ToolInfo])
async def get_agent_tools():
    """Get all available tools from the shared tool registry."""
    try:
        return _tool_info_list()
    except Exception as e:
        logger.error("AGENT_TOOLS: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_name}/test")
async def test_tool(
        tool_name: str,
        body: ToolTestRequest,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """Test a specific tool directly.

    Single-arg tools: pass plain string as `input_data`.
    Multi-arg tools: pass JSON string, e.g. `{"symbol": "AAPL", "period": "1y"}`.
    """
    user_id = (user or {}).get("user_id", "anonymous")
    start_time = time.time()
    log_user_action(logger, "AGENT_TOOL_TEST", user_id, tool=tool_name, input_len=len(body.input_data))

    fn = get_tool_registry().get(tool_name)
    if not fn:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available: {list(get_tool_registry())}",
        )

    try:
        params = list(inspect.signature(fn).parameters.keys())
        if len(params) == 1:
            kwargs = {params[0]: body.input_data}
        else:
            try:
                kwargs = _json.loads(body.input_data)
            except _json.JSONDecodeError:
                raise HTTPException(
                    status_code=422,
                    detail=f"'{tool_name}' requires JSON input with keys: {params}",
                )
        result = fn(**kwargs)
        duration_ms = int((time.time() - start_time) * 1000)
        log_performance_metric(logger, "AGENT_TOOL_TEST", duration_ms, tool=tool_name)
        return {"tool": tool_name, "input": body.input_data, "result": result, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AGENT_TOOL_TEST: failed | tool=%s | error=%s", tool_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
        request: AgentQueryRequest,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """Execute an agent workflow.

    **orchestrator_type**: `autogen` | `custom` | `mcp` | `crewai`

    **workflow** (autogen / custom / crewai):
    - `smart_assistant`      — ToolSelector → ToolExecutor → Summarizer
    - `smart_travel_planner` — TravelToolSelector → ToolExecutor → TravelPlanner
    - `debate`               — Advocate / Critic / Moderator
    - `research`             — 6-agent research pipeline
    - `prompt_evaluation`    — PromptParser → CriteriaJudge → Improver → EvalReporter

    **workflow** (mcp):
    - `smart_assistant` only
    """
    start_time = time.time()
    user_id = (user or {}).get("user_id", "anonymous")
    response = None
    error_msg = None

    log_user_action(
        logger, "AGENT_QUERY_START", user_id,
        orchestrator=request.orchestrator_type,
        workflow=request.workflow,
        question_len=len(request.question),
        tools_requested=request.tools or "all",
        max_steps=request.max_steps,
    )

    try:
        orchestrator = _create_orchestrator(request.orchestrator_type)
        agent_request = AgentRequest(
            question=request.question,
            workflow=request.workflow,
            tools=request.tools,
            max_steps=request.max_steps,
            temperature=request.temperature,
            provider=request.provider,
            conversation_id=request.conversation_id,
            debug=request.debug,
        )
        response = await orchestrator.process_request(agent_request, user)
        logger.info(
            "AGENT_QUERY: done | orchestrator=%s | workflow=%s | tools_used=%s | steps=%d | answer_len=%d",
            request.orchestrator_type, request.workflow,
            response.tools_used, len(response.steps), len(response.answer),
        )
    except Exception as e:
        error_msg = str(e)
        logger.error("AGENT_QUERY: failed | user=%s | error=%s", user_id, e, exc_info=True)

    processing_time_ms = int((time.time() - start_time) * 1000)
    log_performance_metric(
        logger, "AGENT_QUERY", processing_time_ms,
        user_id=user_id,
        orchestrator=request.orchestrator_type,
        workflow=request.workflow,
        tools_used=response.tools_used if response else [],
        steps=len(response.steps) if response else 0,
        success=error_msg is None,
    )

    # --- Persist conversation ---
    conversation_id = request.conversation_id
    try:
        conv_manager = get_container().get_conversation_manager()
        if not conversation_id:
            conversation_id = await conv_manager.create_conversation(
                user_id=user_id,
                chat_type="agent",
                title=request.question[:50] + ("..." if len(request.question) > 50 else ""),
            )
        await conv_manager.add_message(
            conversation_id=conversation_id, speaker="user",
            content=request.question, chat_type="agent",
            extra={"orchestrator_type": request.orchestrator_type, "workflow_type": request.workflow},
        )
        await conv_manager.add_message(
            conversation_id=conversation_id, speaker="assistant",
            content=response.answer if response else f"Error: {error_msg}",
            chat_type="agent",
            extra={
                "orchestrator_type": request.orchestrator_type,
                "workflow_type": request.workflow,
                "tools_used": response.tools_used if response else [],
                "steps": response.steps if response else [],
                "processing_time_ms": processing_time_ms,
                "error_message": error_msg,
            },
        )
    except Exception as save_err:
        logger.warning("AGENT_CONV: failed to save (non-fatal) | error=%s", save_err)

    if error_msg and response is None:
        log_security_event(logger, "AGENT_QUERY_FAILED", user_id, error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    log_user_action(
        logger, "AGENT_QUERY_SUCCESS", user_id,
        conversation_id=conversation_id,
        tools_used=response.tools_used,
        steps=len(response.steps),
        processing_time_ms=processing_time_ms,
    )

    return AgentQueryResponse(
        answer=response.answer,
        steps=response.steps,
        tools_used=response.tools_used,
        available_tools=orchestrator.get_available_tools(),
        available_workflows=orchestrator.get_available_workflows(),
        orchestrator_type=request.orchestrator_type,
        conversation_id=conversation_id,
        debug_info=response.debug_info,
    )


@router.get("/conversations/{conversation_id}/messages")
async def get_agent_conversation_messages(
        conversation_id: str,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """Get full message history for an agent conversation."""
    user_id = (user or {}).get("user_id", "anonymous")
    try:
        conv_manager = get_container().get_conversation_manager()
        messages = await conv_manager.get_messages(conversation_id, user_id)
        return {"conversation_id": conversation_id, "messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error("AGENT_CONV_HISTORY: failed | conversation_id=%s | error=%s", conversation_id, e)
        raise HTTPException(status_code=500, detail=str(e))
