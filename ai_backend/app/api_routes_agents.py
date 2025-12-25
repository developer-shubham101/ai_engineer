"""Agent API routes for LangChain agent workflows."""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user_optional
from app.modules.agents.interfaces import AgentRequest
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


class AgentQueryRequest(BaseModel):
    """Agent query request model."""
    question: str
    tools: List[str] = []  # Specific tools to enable, empty = all available
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


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get agent system status and available tools."""
    try:
        orchestrator = get_agent_orchestrator()

        # Get tool information
        tool_info = []
        for tool_name in orchestrator.get_available_tools():
            tool = orchestrator.tools.get(tool_name)
            if tool:
                tool_info.append(ToolInfo(
                    name=tool.name,
                    description=tool.description
                ))

        return AgentStatusResponse(
            available_tools=tool_info,
            max_steps=orchestrator.max_steps,
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
    
    This endpoint provides a sandbox environment for LangChain agent experimentation.
    
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
    
    **Example Queries:**
    - "What is the status of my tickets?"
    - "Search for vacation policy documents"
    - "Analyze user engagement data"
    - "Generate performance research data"
    """
    try:
        # Get orchestrator based on request type
        # We bypass the default container.get_agent_orchestrator() to support dynamic selection
        from app.modules.agents.factories import AgentOrchestratorFactory
        container = get_container()
        orchestrator = AgentOrchestratorFactory.create_orchestrator(
            orchestrator_type=request.orchestrator_type,
            vector_store=container.get_vector_store()
        )

        # Create agent request
        agent_request = AgentRequest(
            question=request.question,
            tools=request.tools,
            max_steps=request.max_steps,
            temperature=request.temperature,
            provider=request.provider,
            conversation_id=request.conversation_id,
            debug=request.debug
        )

        # Process request
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
    """List all available agent tools."""
    try:
        orchestrator = get_agent_orchestrator()

        tools = []
        for tool_name in orchestrator.get_available_tools():
            tool = orchestrator.tools.get(tool_name)
            if tool:
                tools.append(ToolInfo(
                    name=tool.name,
                    description=tool.description
                ))

        return tools
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_name}/test")
async def test_tool(
        tool_name: str,
        input_data: str,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Test a specific tool directly.
    
    **For Research and Development:**
    This endpoint allows direct tool testing without agent orchestration.
    Useful for debugging and tool development.
    """
    try:
        orchestrator = get_agent_orchestrator()

        tool = orchestrator.tools.get(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        context = {"user": user or {}}
        result = await tool.execute(input_data, context)

        return {
            "tool": tool_name,
            "input": input_data,
            "result": result,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
