"""CrewAI interfaces for multi-agent workflows."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class CrewRequest(BaseModel):
    """CrewAI request model."""
    topic: str
    workflow_type: str = "debate"  # debate, research, analysis
    max_iterations: int = 3
    temperature: float = 0.7
    provider: str = "local"
    conversation_id: Optional[str] = None


class CrewResponse(BaseModel):
    """CrewAI response model."""
    result: str
    workflow_type: str
    agents_used: List[str]
    iterations: int
    execution_time_ms: int
    debug_info: Optional[Dict[str, Any]] = None


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