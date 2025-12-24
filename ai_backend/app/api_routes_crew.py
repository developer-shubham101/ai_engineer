"""CrewAI API routes for multi-agent workflows."""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user_optional
from app.modules.crew_ai.interfaces import CrewRequest, CrewResponse
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/crew", tags=["CrewAI"])


class CrewQueryRequest(BaseModel):
    """CrewAI query request model."""
    topic: str
    workflow_type: str = "debate"  # debate, research, analysis
    max_iterations: int = 3
    temperature: float = 0.7
    provider: str = "local"
    conversation_id: Optional[str] = None


class CrewQueryResponse(BaseModel):
    """CrewAI query response model."""
    result: str
    workflow_type: str
    agents_used: List[str]
    iterations: int
    execution_time_ms: int
    available_workflows: List[str]
    debug_info: Optional[Dict[str, Any]] = None


class WorkflowInfo(BaseModel):
    """Workflow information model."""
    name: str
    description: str
    agents: List[str]


class CrewStatusResponse(BaseModel):
    """CrewAI status response model."""
    available_workflows: List[WorkflowInfo]
    status: str


def get_crew_orchestrator():
    """Get CrewAI orchestrator instance from container."""
    container = get_container()
    container.initialize()
    return container.get_crew_orchestrator()


@router.get("/status", response_model=CrewStatusResponse)
async def get_crew_status():
    """Get CrewAI system status and available workflows."""
    try:
        orchestrator = get_crew_orchestrator()
        
        workflow_info = [
            WorkflowInfo(
                name="debate",
                description="Multi-agent debate with Advocate, Critic, and Moderator",
                agents=["Advocate", "Critic", "Moderator"]
            ),
            WorkflowInfo(
                name="research",
                description="Comprehensive research with Researcher, Analyst, and Synthesizer",
                agents=["Researcher", "Analyst", "Synthesizer"]
            ),
            WorkflowInfo(
                name="analysis",
                description="Structured analysis with Examiner and Evaluator",
                agents=["Examiner", "Evaluator"]
            )
        ]
        
        return CrewStatusResponse(
            available_workflows=workflow_info,
            status="active"
        )
    except Exception as e:
        logger.error(f"Failed to get CrewAI status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=CrewQueryResponse)
async def crew_query(
    request: CrewQueryRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Execute CrewAI multi-agent workflow using official CrewAI library.
    
    **Available Workflows:**
    
    - **debate**: Multi-agent debate with opposing viewpoints
      - Advocate: Presents arguments in favor
      - Critic: Presents counterarguments  
      - Moderator: Provides balanced synthesis
    
    - **research**: Comprehensive research workflow
      - Researcher: Gathers information and facts
      - Analyst: Provides deep analysis
      - Synthesizer: Creates comprehensive report
    
    **Example Topics:**
    - "Should companies adopt remote work policies?"
    - "Impact of AI on job market"
    - "Benefits of renewable energy adoption"
    - "Effectiveness of agile development methodologies"
    
    **Note**: Now uses official CrewAI library with YAML configuration.
    """
    try:
        orchestrator = get_crew_orchestrator()
        
        # Create crew request
        crew_request = CrewRequest(
            topic=request.topic,
            workflow_type=request.workflow_type,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            provider=request.provider,
            conversation_id=request.conversation_id
        )
        
        # Execute workflow
        response = await orchestrator.execute_workflow(crew_request, user)
        
        return CrewQueryResponse(
            result=response.result,
            workflow_type=response.workflow_type,
            agents_used=response.agents_used,
            iterations=response.iterations,
            execution_time_ms=response.execution_time_ms,
            available_workflows=orchestrator.get_available_workflows(),
            debug_info=response.debug_info
        )
        
    except Exception as e:
        logger.error(f"CrewAI query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows", response_model=List[WorkflowInfo])
async def list_workflows():
    """List all available CrewAI workflows."""
    try:
        return [
            WorkflowInfo(
                name="debate",
                description="Multi-agent debate with opposing viewpoints",
                agents=["Advocate", "Critic", "Moderator"]
            ),
            WorkflowInfo(
                name="research",
                description="Comprehensive research with multiple perspectives",
                agents=["Researcher", "Analyst", "Synthesizer"]
            ),
            WorkflowInfo(
                name="analysis",
                description="Structured analysis and evaluation",
                agents=["Examiner", "Evaluator"]
            )
        ]
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))