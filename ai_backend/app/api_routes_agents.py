"""Agent API routes."""

import logging
import time
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user_optional
from app.modules.agents.interfaces import AgentRequest
from app.modules.agents.agent_runner import REGISTRY, call_tool
from app.modules.integration import get_container
from app.logging_config import log_user_action, log_performance_metric, log_security_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


class AgentQueryRequest(BaseModel):
    """Agent query request model."""
    question: str
    tools: List[str] = []
    max_steps: int = 5
    temperature: float = 0.1
    provider: str = "local"
    orchestrator_type: str = "custom"
    conversation_id: Optional[str] = None
    debug: bool = False


class AgentQueryResponse(BaseModel):
    """Agent query response model."""
    answer: str
    steps: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    available_tools: List[str] = []
    conversation_id: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


class ToolTestRequest(BaseModel):
    """Tool test request body."""
    input_data: str


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str


class AgentStatusResponse(BaseModel):
    """Agent status response model."""
    available_tools: List[ToolInfo]
    max_steps: int
    status: str


def get_agent_orchestrator():
    """Get agent orchestrator instance from container."""
    container = get_container()
    container.initialize()
    return container.get_agent_orchestrator()


def _get_all_tool_info() -> List[ToolInfo]:
    """
    Build unified ToolInfo list from both sources:
    - Custom orchestrator .tools dict  (ITool-based)
    - agent_runner REGISTRY            (function-based)
    """
    seen = set()
    tools = []

    try:
        orchestrator = get_agent_orchestrator()
        if hasattr(orchestrator, "tools"):
            for name, tool in orchestrator.tools.items():
                if name not in seen:
                    tools.append(ToolInfo(name=tool.name, description=tool.description))
                    seen.add(name)
        logger.debug("AGENT_TOOLS: loaded %d tools from orchestrator", len(tools))
    except Exception as e:
        logger.warning("AGENT_TOOLS: could not load orchestrator tools: %s", e)

    registry_count = 0
    for name, meta in REGISTRY.items():
        if name not in seen:
            tools.append(ToolInfo(name=name, description=meta["description"]))
            seen.add(name)
            registry_count += 1

    logger.debug("AGENT_TOOLS: loaded %d tools from REGISTRY, total=%d", registry_count, len(tools))
    return tools


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get agent system status and available tools."""
    logger.debug("AGENT_STATUS: request received")
    try:
        tool_info = _get_all_tool_info()
        orchestrator = get_agent_orchestrator()
        max_steps = getattr(orchestrator, "max_steps", 5)

        logger.info("AGENT_STATUS: active | tools=%d | max_steps=%d", len(tool_info), max_steps)
        return AgentStatusResponse(
            available_tools=tool_info,
            max_steps=max_steps,
            status="active"
        )
    except Exception as e:
        logger.error("AGENT_STATUS: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
        request: AgentQueryRequest,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Execute agent workflow with tools.

    Saves the conversation (user question + agent answer + steps + tools used)
    to the agent_messages table — separate from RAG /query conversations.

    **Available Tools:**
    - `search_documents`: Search company knowledge base
    - `get_user_tickets`: Get user support tickets (mock data)
    - `get_ticket_comments`: Get ticket conversation history
    - `analyze_data`: Analyze data patterns and statistics
    - `research_data`: Generate research metrics
    - `summarize_status`: Summarize information
    - `web_search`: Search the internet for real-time information
    - `scrape_url`: Fetch and extract full content from a URL
    - `get_stock_price`: Get real-time stock price
    - `get_weather`: Get current weather for a city
    - `save_text_file`: Save text content to a file
    """
    start_time = time.time()
    error_msg = None
    response = None
    user_id = (user or {}).get("user_id", "anonymous")
    user_role = (user or {}).get("role", "none")

    log_user_action(
        logger, "AGENT_QUERY_START", user_id,
        orchestrator=request.orchestrator_type,
        question_len=len(request.question),
        tools_requested=request.tools or "all",
        max_steps=request.max_steps,
        temperature=request.temperature,
        conversation_id=request.conversation_id or "new"
    )
    logger.debug("AGENT_QUERY: question=%r | user=%s | role=%s", request.question[:100], user_id, user_role)

    try:
        from app.modules.agents.factories import AgentOrchestratorFactory
        container = get_container()

        logger.debug("AGENT_QUERY: creating orchestrator type=%s", request.orchestrator_type)
        orchestrator = AgentOrchestratorFactory.create_orchestrator(
            orchestrator_type=request.orchestrator_type,
            vector_store=container.get_vector_store()
        )
        logger.debug("AGENT_QUERY: orchestrator ready | available_tools=%s", orchestrator.get_available_tools())

        agent_request = AgentRequest(
            question=request.question,
            tools=request.tools,
            max_steps=request.max_steps,
            temperature=request.temperature,
            provider=request.provider,
            conversation_id=request.conversation_id,
            debug=request.debug
        )

        logger.debug("AGENT_QUERY: dispatching to orchestrator")
        response = await orchestrator.process_request(agent_request, user)

        logger.info(
            "AGENT_QUERY: completed | tools_used=%s | steps=%d | answer_len=%d",
            response.tools_used, len(response.steps), len(response.answer)
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("AGENT_QUERY: orchestrator failed | user=%s | error=%s", user_id, e, exc_info=True)

    processing_time_ms = int((time.time() - start_time) * 1000)
    log_performance_metric(
        logger, "AGENT_QUERY", processing_time_ms,
        user_id=user_id,
        orchestrator=request.orchestrator_type,
        tools_used=response.tools_used if response else [],
        steps=len(response.steps) if response else 0,
        success=error_msg is None
    )

    # --- Save to agent_messages table ---
    conversation_id = request.conversation_id
    try:
        conv_manager = get_container().get_conversation_manager()

        if not conversation_id:
            conversation_id = await conv_manager.create_conversation(
                user_id=user_id,
                chat_type="agent",
                title=request.question[:50] + ("..." if len(request.question) > 50 else "")
            )
            logger.debug("AGENT_CONV: auto-created conversation_id=%s for user=%s", conversation_id, user_id)
        else:
            logger.debug("AGENT_CONV: using existing conversation_id=%s", conversation_id)

        await conv_manager.add_message(
            conversation_id=conversation_id,
            speaker="user",
            content=request.question,
            chat_type="agent",
            extra={
                "user_query": request.question,
                "orchestrator_type": request.orchestrator_type,
            }
        )

        await conv_manager.add_message(
            conversation_id=conversation_id,
            speaker="assistant",
            content=response.answer if response else f"Error: {error_msg}",
            chat_type="agent",
            extra={
                "user_query": request.question,
                "tools_used": response.tools_used if response else [],
                "steps": response.steps if response else [],
                "orchestrator_type": request.orchestrator_type,
                "processing_time_ms": processing_time_ms,
                "error_message": error_msg
            }
        )
        logger.debug("AGENT_CONV: saved 2 messages to conversation_id=%s", conversation_id)

    except Exception as save_err:
        logger.warning("AGENT_CONV: failed to save conversation (non-fatal) | error=%s", save_err)

    if error_msg and response is None:
        log_security_event(logger, "AGENT_QUERY_FAILED", user_id, error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    log_user_action(
        logger, "AGENT_QUERY_SUCCESS", user_id,
        conversation_id=conversation_id,
        tools_used=response.tools_used,
        steps=len(response.steps),
        processing_time_ms=processing_time_ms
    )

    return AgentQueryResponse(
        answer=response.answer,
        steps=response.steps,
        tools_used=response.tools_used,
        available_tools=orchestrator.get_available_tools(),
        conversation_id=conversation_id,
        debug_info=response.debug_info
    )


@router.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List all available agent tools (orchestrator tools + function-based registry tools)."""
    logger.debug("AGENT_TOOLS_LIST: request received")
    try:
        tools = _get_all_tool_info()
        logger.info("AGENT_TOOLS_LIST: returning %d tools", len(tools))
        return tools
    except Exception as e:
        logger.error("AGENT_TOOLS_LIST: failed | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_name}/test")
async def test_tool(
        tool_name: str,
        body: ToolTestRequest,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Test a specific tool directly.

    Accepts input as a JSON body: `{"input_data": "your input here"}`

    Supports both orchestrator tools (search_documents, get_user_tickets, etc.)
    and function-based tools (web_search, scrape_url, get_stock_price, get_weather, save_text_file).
    """
    user_id = (user or {}).get("user_id", "anonymous")
    start_time = time.time()

    log_user_action(logger, "AGENT_TOOL_TEST", user_id, tool=tool_name, input_len=len(body.input_data))
    logger.debug("AGENT_TOOL_TEST: tool=%s | input=%r", tool_name, body.input_data[:100])

    try:
        context = {"user": user or {}}

        # 1. Try orchestrator tools first (ITool interface — async execute)
        try:
            orchestrator = get_agent_orchestrator()
            if hasattr(orchestrator, "tools"):
                tool = orchestrator.tools.get(tool_name)
                if tool:
                    logger.debug("AGENT_TOOL_TEST: found in orchestrator, executing")
                    result = await tool.execute(body.input_data, context)
                    duration_ms = int((time.time() - start_time) * 1000)
                    log_performance_metric(logger, "AGENT_TOOL_TEST", duration_ms, tool=tool_name, source="orchestrator")
                    logger.info("AGENT_TOOL_TEST: success | tool=%s | source=orchestrator | duration_ms=%d", tool_name, duration_ms)
                    return {
                        "tool": tool_name,
                        "input": body.input_data,
                        "result": result,
                        "status": "success",
                        "source": "orchestrator"
                    }
        except Exception as e:
            logger.debug("AGENT_TOOL_TEST: orchestrator lookup failed, trying registry | error=%s", e)

        # 2. Fall back to REGISTRY (function-based tools)
        if tool_name in REGISTRY:
            meta = REGISTRY[tool_name]
            first_arg = meta["args"][0]
            logger.debug("AGENT_TOOL_TEST: found in REGISTRY, executing | arg=%s", first_arg)
            result = call_tool(tool_name, {first_arg: body.input_data})
            duration_ms = int((time.time() - start_time) * 1000)
            log_performance_metric(logger, "AGENT_TOOL_TEST", duration_ms, tool=tool_name, source="registry")
            logger.info("AGENT_TOOL_TEST: success | tool=%s | source=registry | duration_ms=%d", tool_name, duration_ms)
            return {
                "tool": tool_name,
                "input": body.input_data,
                "result": result,
                "status": "success",
                "source": "registry"
            }

        all_tools = [t.name for t in _get_all_tool_info()]
        logger.warning("AGENT_TOOL_TEST: tool not found | tool=%s | available=%s", tool_name, all_tools)
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available tools: {all_tools}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("AGENT_TOOL_TEST: failed | tool=%s | error=%s", tool_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages")
async def get_agent_conversation_messages(
        conversation_id: str,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Get agent conversation messages from agent_messages table.

    Returns the full history of a specific agent conversation including
    steps taken and tools used per turn.
    """
    user_id = (user or {}).get("user_id", "anonymous")
    logger.debug("AGENT_CONV_HISTORY: request | conversation_id=%s | user=%s", conversation_id, user_id)

    try:
        conv_manager = get_container().get_conversation_manager()
        messages = await conv_manager.get_messages(conversation_id, user_id)

        logger.info("AGENT_CONV_HISTORY: returned %d messages | conversation_id=%s | user=%s",
                    len(messages), conversation_id, user_id)
        return {"conversation_id": conversation_id, "messages": messages, "count": len(messages)}

    except Exception as e:
        logger.error("AGENT_CONV_HISTORY: failed | conversation_id=%s | error=%s", conversation_id, e)
        raise HTTPException(status_code=500, detail=str(e))
