"""Dynamic tool implementations for research and experimentation."""

import logging
from typing import Dict, Any
from .interfaces import ITool
from .utils import MockDataProvider, AnalysisProvider, DocumentFormatter
from ..vector_db.interfaces import IVectorStore

logger = logging.getLogger(__name__)


class SearchDocumentsTool(ITool):
    """Search documents using vector store."""
    
    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        return "search_documents"
    
    @property
    def description(self) -> str:
        return "Search company documents and knowledge base. Input: search query string"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        try:
            results = await self.vector_store.search_documents(
                query=input_data.strip(),
                top_k=3
            )
            return DocumentFormatter.format_search_results(results)
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"Error searching documents: {str(e)}"


class GetUserTicketsTool(ITool):
    """Get user support tickets (mock implementation for research)."""
    
    @property
    def name(self) -> str:
        return "get_user_tickets"
    
    @property
    def description(self) -> str:
        return "Get user's support tickets. Input: user_id or 'current' for current user"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        user = context.get("user", {})
        user_id = user.get("user_id", "unknown")
        
        mock_tickets = MockDataProvider.get_mock_tickets(user_id)
        
        result = f"Found {len(mock_tickets)} tickets for user {user_id}:\n"
        for ticket in mock_tickets:
            result += f"- {ticket['id']}: {ticket['title']} ({ticket['status']})\n"
        
        return result


class GetTicketCommentsTool(ITool):
    """Get ticket comments and history."""
    
    @property
    def name(self) -> str:
        return "get_ticket_comments"
    
    @property
    def description(self) -> str:
        return "Get comments for a specific ticket. Input: ticket_id (e.g., TKT-001)"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        ticket_id = input_data.strip()
        
        comments = MockDataProvider.get_mock_comments(ticket_id)
        if not comments:
            return f"No comments found for ticket {ticket_id}"
        
        result = f"Comments for {ticket_id}:\n"
        for comment in comments:
            result += f"[{comment['date']}] {comment['author']}: {comment['text']}\n"
        
        return result


class AnalyzeDataTool(ITool):
    """Analyze data for research purposes."""
    
    @property
    def name(self) -> str:
        return "analyze_data"
    
    @property
    def description(self) -> str:
        return "Analyze data patterns or statistics. Input: data description or query"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        return AnalysisProvider.analyze_query(input_data)


class SummarizeStatusTool(ITool):
    """Summarize status information."""
    
    @property
    def name(self) -> str:
        return "summarize_status"
    
    @property
    def description(self) -> str:
        return "Summarize status information. Input: data to summarize"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        return f"Summary: {input_data[:200]}... [Status compiled from available information]"


class ResearchDataTool(ITool):
    """Generate research data for experimentation."""
    
    @property
    def name(self) -> str:
        return "research_data"
    
    @property
    def description(self) -> str:
        return "Generate research data or metrics. Input: data type (users, performance, trends, etc.)"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        data_type = input_data.strip().lower()
        return MockDataProvider.get_research_data(data_type)


class WebSearchTool(ITool):
    """Search the internet for real-time information."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the internet for real-time information. Use for current events, facts, or anything not in internal documents. Input: search query string"

    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        from ..function_tools.tool_web_search import web_search
        result = web_search(input_data.strip())
        if result.get("status") == "success":
            return f"Web search results for '{input_data}':\n\n{result['formatted']}"
        return f"Web search failed: {result.get('error', 'Unknown error')}"


class ScrapeUrlTool(ITool):
    """Fetch and extract full text content from a URL."""

    @property
    def name(self) -> str:
        return "scrape_url"

    @property
    def description(self) -> str:
        return "Fetch and extract full text content from a URL. Use after web_search to get detailed information. Input: URL string"

    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        from ..function_tools.tool_web_scraper import scrape_url
        result = scrape_url(input_data.strip())
        if result.get("status") == "success":
            truncated_note = " (truncated)" if result.get("truncated") else ""
            return f"Content from {input_data}{truncated_note}:\n\n{result['content']}"
        return f"Scrape failed: {result.get('error', 'Unknown error')}"


# Deprecated: Use ToolFactory.create_default_tools() instead
# This function is kept for backward compatibility
def create_default_tools(vector_store=None):
    """Create default tool set for agents. DEPRECATED: Use ToolFactory instead."""
    from .factories import ToolFactory
    return ToolFactory.create_default_tools(vector_store)