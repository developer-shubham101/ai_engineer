"""Factory for creating agent tools and orchestrators."""

from typing import List, Optional
from .interfaces import ITool, IAgentOrchestrator
from .orchestrators import CustomOrchestrator, AutoGenOrchestrator, AUTOGEN_AVAILABLE
from .tools import (
    SearchDocumentsTool, GetUserTicketsTool, GetTicketCommentsTool,
    AnalyzeDataTool, SummarizeStatusTool, ResearchDataTool
)
from ..vector_db.interfaces import IVectorStore


class ToolFactory:
    """Factory for creating agent tools."""
    
    @staticmethod
    def create_search_tool(vector_store: IVectorStore) -> ITool:
        """Create search documents tool."""
        return SearchDocumentsTool(vector_store)
    
    @staticmethod
    def create_ticket_tool() -> ITool:
        """Create user tickets tool."""
        return GetUserTicketsTool()
    
    @staticmethod
    def create_comments_tool() -> ITool:
        """Create ticket comments tool."""
        return GetTicketCommentsTool()
    
    @staticmethod
    def create_analysis_tool() -> ITool:
        """Create data analysis tool."""
        return AnalyzeDataTool()
    
    @staticmethod
    def create_summary_tool() -> ITool:
        """Create summary tool."""
        return SummarizeStatusTool()
    
    @staticmethod
    def create_research_tool() -> ITool:
        """Create research data tool."""
        return ResearchDataTool()
    
    @classmethod
    def create_default_tools(cls, vector_store: Optional[IVectorStore] = None) -> List[ITool]:
        """Create default tool set."""
        tools = [
            cls.create_ticket_tool(),
            cls.create_comments_tool(),
            cls.create_analysis_tool(),
            cls.create_summary_tool(),
            cls.create_research_tool()
        ]
        
        if vector_store:
            tools.append(cls.create_search_tool(vector_store))
        
        return tools


class AgentOrchestratorFactory:
    """Factory for creating agent orchestrators."""
    
    @staticmethod
    def create_orchestrator(
        orchestrator_type: str = "autogen",
        vector_store: Optional[IVectorStore] = None,
        tools: Optional[List[ITool]] = None
    ) -> IAgentOrchestrator:
        """Create agent orchestrator based on type."""
        if orchestrator_type.lower() == "autogen":
            if not AUTOGEN_AVAILABLE:
                print("AutoGen not available, using custom orchestrator")
                orchestrator_type = "custom"
            else:
                return AutoGenOrchestrator()
        
        if orchestrator_type.lower() == "custom":
            if tools is None:
                tools = ToolFactory.create_default_tools(vector_store)
            
            orchestrator = CustomOrchestrator()
            
            for tool in tools:
                orchestrator.register_tool(tool)
            
            return orchestrator
        
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")
    
    @staticmethod
    def get_available_types() -> dict:
        """Get available orchestrator types."""
        return {
            "custom": True,
            "autogen": AUTOGEN_AVAILABLE
        }