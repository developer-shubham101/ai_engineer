"""Factory for creating agent tools and orchestrators."""

import logging
from typing import List, Optional

from .interfaces import ITool, IAgentOrchestrator
from .orchestrators import (
    CustomOrchestrator, AutoGenOrchestrator, AUTOGEN_AVAILABLE,
    MCPOrchestrator, MCP_AVAILABLE,
    CrewAIOrchestrator, CREWAI_AVAILABLE,
)
from .tools import (
    SearchDocumentsTool, GetUserTicketsTool, GetTicketCommentsTool,
    AnalyzeDataTool, SummarizeStatusTool, ResearchDataTool,
    WebSearchTool, ScrapeUrlTool
)
from ..vector_db.interfaces import IVectorStore

logger = logging.getLogger(__name__)


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

    @staticmethod
    def create_web_search_tool() -> ITool:
        """Create internet web search tool."""
        return WebSearchTool()

    @staticmethod
    def create_scrape_url_tool() -> ITool:
        """Create URL scraper tool."""
        return ScrapeUrlTool()

    @classmethod
    def create_default_tools(cls, vector_store: Optional[IVectorStore] = None) -> List[ITool]:
        """Create default tool set."""
        tools = [
            cls.create_ticket_tool(),
            cls.create_comments_tool(),
            cls.create_analysis_tool(),
            cls.create_summary_tool(),
            cls.create_research_tool(),
            cls.create_web_search_tool(),
            cls.create_scrape_url_tool(),
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
            tools: Optional[List[ITool]] = None,
            colabllm_config: Optional[dict] = None
    ) -> IAgentOrchestrator:
        """Create agent orchestrator based on type."""
        if orchestrator_type.lower() == "autogen":
            if not AUTOGEN_AVAILABLE:
                logger.warning("AutoGen not available, using custom orchestrator")
                orchestrator_type = "custom"
            else:
                # Create LlamaServerProvider and use its client
                from ..llm.providers.llamaserver import LlamaServerProvider
                provider = LlamaServerProvider(colabllm_config or {})
                return AutoGenOrchestrator(model_client=provider.client)

        if orchestrator_type.lower() == "custom":
            from ..llm.providers.llamaserver import LlamaServerProvider
            from ..llm.interfaces import ILLMProvider
            import asyncio

            provider: ILLMProvider = LlamaServerProvider(colabllm_config or {})

            async def llm_fn(system: str, user: str) -> str:
                return await asyncio.to_thread(
                    provider.generate, f"{system}\n\n{user}"
                )

            return CustomOrchestrator(llm_fn=llm_fn)

        if orchestrator_type.lower() == "mcp":
            if not MCP_AVAILABLE:
                raise ValueError("MCPOrchestrator not available (autogen not installed)")
            from ..llm.providers.llamaserver import LlamaServerProvider
            provider = LlamaServerProvider(colabllm_config or {})
            return MCPOrchestrator(model_client=provider.client)

        if orchestrator_type.lower() == "crewai":
            if not CREWAI_AVAILABLE:
                raise ValueError("CrewAIOrchestrator not available (crewai not installed)")
            return CrewAIOrchestrator()

        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")

    @staticmethod
    def get_available_types() -> dict:
        """Get available orchestrator types."""
        return {
            "custom": True,
            "autogen": AUTOGEN_AVAILABLE,
            "mcp": MCP_AVAILABLE,
            "crewai": CREWAI_AVAILABLE,
        }
