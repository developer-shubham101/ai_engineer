"""Agent API routes for LangChain agent workflows."""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user_optional
from app.modules.agents.interfaces import AgentRequest
from app.modules.agents.agent_runner import REGISTRY, call_tool
from app.modules.integration import get_container

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
    - Custom orchestrator .tools dict  (ITool-based: search_documents, tickets, etc.)
    - agent_runner REGISTRY            (function-based: web_search, scrape_url, stock, weather, file)
    """
    seen = set()
    tools = []

    # 1. Orchestrator tools (ITool interface)
    try:
        orchestrator = get_agent_orchestrator()
        if hasattr(orchestrator, "tools"):
            for name, tool in orchestrator.tools.items():
                if name not in seen:
                    tools.append(ToolInfo(name=tool.name, description=tool.description))
                    seen.add(name)
    except Exception as e:
        logger.warning(f"Could not load orchestrator tools: {e}")

    # 2. Function-based tools from REGISTRY
    for name, meta in REGISTRY.items():
        if name not in seen:
            tools.append(ToolInfo(name=name, description=meta["description"]))
            seen.add(name)

    return tools


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get agent system status and available tools."""
    try:
        tool_info = _get_all_tool_info()
        orchestrator = get_agent_orchestrator()
        max_steps = getattr(orchestrator, "max_steps", 5)

        return AgentStatusResponse(
            available_tools=tool_info,
            max_steps=max_steps,
            status="active"
        )
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
        request: AgentQueryRequest,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Execute agent workflow with tools.

    **Safety Features:**
    - Hard step limit (max 5 steps)
    - Tool whitelisting (only pre-approved tools)
    - No direct database access
    - Sandboxed execution

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
    try:
        from app.modules.agents.factories import AgentOrchestratorFactory
        container = get_container()
        orchestrator = AgentOrchestratorFactory.create_orchestrator(
            orchestrator_type=request.orchestrator_type,
            vector_store=container.get_vector_store()
        )

        agent_request = AgentRequest(
            question=request.question,
            tools=request.tools,
            max_steps=request.max_steps,
            temperature=request.temperature,
            provider=request.provider,
            conversation_id=request.conversation_id,
            debug=request.debug
        )

        response = await orchestrator.process_request(agent_request, user)

        return AgentQueryResponse(
            answer=response.answer,
            steps=response.steps,
            tools_used=response.tools_used,
            available_tools=orchestrator.get_available_tools(),
            debug_info=response.debug_info
        )

    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List all available agent tools (orchestrator tools + function-based registry tools)."""
    try:
        return _get_all_tool_info()
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
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
    try:
        context = {"user": user or {}}

        # 1. Try orchestrator tools first (ITool interface — async execute)
        try:
            orchestrator = get_agent_orchestrator()
            if hasattr(orchestrator, "tools"):
                tool = orchestrator.tools.get(tool_name)
                if tool:
                    result = await tool.execute(body.input_data, context)
                    return {
                        "tool": tool_name,
                        "input": body.input_data,
                        "result": result,
                        "status": "success",
                        "source": "orchestrator"
                    }
        except Exception as e:
            logger.debug(f"Orchestrator tool lookup failed, trying registry: {e}")

        # 2. Fall back to REGISTRY (function-based tools)
        if tool_name in REGISTRY:
            meta = REGISTRY[tool_name]
            # Pass input_data as the first positional arg
            first_arg = meta["args"][0]
            result = call_tool(tool_name, {first_arg: body.input_data})
            return {
                "tool": tool_name,
                "input": body.input_data,
                "result": result,
                "status": "success",
                "source": "registry"
            }

        all_tools = [t.name for t in _get_all_tool_info()]
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available tools: {all_tools}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
