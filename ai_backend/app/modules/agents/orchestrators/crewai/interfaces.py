"""CrewAI interfaces for multi-agent workflows."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class CrewRequest(BaseModel):
    """CrewAI request model."""
    topic: str = Field(..., description="The topic or question for the crew to work on")
    workflow_type: str = Field(default="debate", description="Workflow type: debate, research, analysis, smart_assistant, smart_travel_planner, prompt_evaluation")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum iterations for the workflow")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    provider: str = Field(default="local", description="LLM provider")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for tracking")


class CrewResponse(BaseModel):
    """CrewAI response model."""
    result: str = Field(..., description="The final result from the crew workflow")
    workflow_type: str = Field(..., description="The workflow type that was executed")
    agents_used: List[str] = Field(default_factory=list, description="List of agents that participated")
    iterations: int = Field(default=0, description="Number of iterations completed")
    execution_time_ms: int = Field(default=0, description="Execution time in milliseconds")
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="Additional debug information")


class ICrewOrchestrator(ABC):
    """Interface for CrewAI orchestration."""
    
    @abstractmethod
    async def execute_workflow(self, request: CrewRequest, user: Optional[Dict[str, Any]] = None) -> CrewResponse:
        """Execute CrewAI workflow."""
        pass
    
    @abstractmethod
    def get_available_workflows(self) -> List[str]:
        """Get available workflow types."""
        pass