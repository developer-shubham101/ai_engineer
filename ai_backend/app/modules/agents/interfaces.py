"""Agent interfaces for dynamic tool system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Agent request model."""
    question: str
    workflow: str = "debate"  # Workflow to execute: debate, research, etc.
    tools: List[str] = []  # Tool names to enable (empty = all available)
    max_steps: int = 5
    temperature: float = 0.1
    provider: str = "local"
    conversation_id: Optional[str] = None
    debug: bool = False


class AgentResponse(BaseModel):
    """Agent response model."""
    answer: str
    steps: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    final_step: bool = True
    debug_info: Optional[Dict[str, Any]] = None


class ITool(ABC):
    """Interface for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        """Execute tool with input."""
        pass


class IAgentOrchestrator(ABC):
    """Interface for agent orchestration."""
    
    @abstractmethod
    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process agent request."""
        pass
    
    @abstractmethod
    def register_tool(self, tool: ITool) -> None:
        """Register a tool."""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        pass