"""AutoGen-based agent orchestrator for multi-agent conversations using AutoGen v0.4."""

import logging
from typing import Dict, Any, Optional, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class AutoGenOrchestrator(IAgentOrchestrator):
    """AutoGen-based multi-agent orchestrator using v0.4 API."""

    def __init__(self, model_client):
        # Use injected model client or create default
        self.model_client = model_client

    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process agent request using AutoGen."""
        try:
            query = request.question
            lower_q = query.lower()

            return await self._execute_research_workflow(query)
            # if "research" in lower_q:
            #     return await self._execute_research_workflow(query)
            # else:
            #     return await self._execute_debate_workflow(query)

        except Exception as e:
            logger.error(f"AutoGen workflow failed: {e}", exc_info=True)
            return AgentResponse(
                answer=f"Workflow failed: {str(e)}",
                steps=[],
                tools_used=[],
                final_step=True
            )

    def register_tool(self, tool: Any) -> None:
        """Register a tool (Not implemented for v0.4 team yet)."""
        pass

    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return ["search_internet", "fetch_url", "get_stock", "get_city_weather", "save_file"]

    async def _execute_debate_workflow(self, query: str) -> AgentResponse:
        """Execute debate workflow with AutoGen agents."""

        # 1. Create Agents
        advocate = AssistantAgent(
            name="Advocate",
            system_message="You argue FOR the given topic with strong supporting evidence. Keep responses concise.",
            model_client=self.model_client,
        )

        critic = AssistantAgent(
            name="Critic",
            system_message="You argue AGAINST the given topic with counterarguments. Keep responses concise.",
            model_client=self.model_client,
        )

        moderator = AssistantAgent(
            name="Moderator",
            system_message="You moderate the debate and provide final balanced summary. Keep responses concise.",
            model_client=self.model_client,
        )

        # 2. Define Termination Condition
        termination = MaxMessageTermination(max_messages=4)

        # 3. Create Team
        team = RoundRobinGroupChat(
            participants=[advocate, critic, moderator],
            termination_condition=termination
        )

        # 4. Run Team — single user-turn task
        task_message = f"Debate topic: {query}"
        stream = team.run_stream(task=task_message)

        steps = []
        final_result = ""

        async for message in stream:
            if hasattr(message, 'content') and hasattr(message, 'source'):
                steps.append(f"{message.source}: {str(message.content)[:100]}...")
                final_result = str(message.content)

        return AgentResponse(
            answer=final_result,
            steps=[{"source": "autogen", "content": s} for s in steps],
            tools_used=[],
            final_step=True
        )

    def _build_all_tools(self) -> List:
        """Build all available tool functions for AutoGen agents."""
        from ...function_tools.tool_web_search import web_search
        from ...function_tools.tool_web_scraper import scrape_url
        from ...function_tools.tool_stock import get_stock_price
        from ...function_tools.tool_weather import get_weather
        from ...function_tools.tool_file import save_text_file

        def search_internet(query: str) -> str:
            """Search the internet for real-time information on any topic."""
            result = web_search(query, max_results=5)
            if result.get("status") == "success":
                return result["formatted"]
            return f"Search failed: {result.get('error', 'Unknown error')}"

        def fetch_url(url: str) -> str:
            """Fetch and extract full text content from a URL."""
            result = scrape_url(url)
            if result.get("status") == "success":
                return result["content"]
            return f"Fetch failed: {result.get('error', 'Unknown error')}"

        def get_stock(symbol: str) -> str:
            """Get the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
            result = get_stock_price(symbol)
            if result.get("status") == "success":
                return f"{result['symbol']}: ${result['price']}"
            return f"Stock lookup failed for {symbol}: {result.get('error', 'Unknown error')}"

        def get_city_weather(city: str) -> str:
            """Get current weather conditions for a city."""
            result = get_weather(city)
            if result.get("status") in ("success", "demo_data"):
                return (
                    f"{result['city']}: {result['temperature']}, "
                    f"{result['description']}, humidity {result['humidity']}"
                )
            return f"Weather lookup failed for {city}: {result.get('error', 'Unknown error')}"

        def save_file(filename: str, content: str) -> str:
            """Save text content to a file in user_uploaded_files/."""
            result = save_text_file(filename, content)
            if result.get("status") == "success":
                return f"Saved '{result['filename']}' ({result['size']} chars) at {result['filepath']}"
            return f"Save failed: {result.get('error', 'Unknown error')}"

        return [search_internet, fetch_url, get_stock, get_city_weather, save_file]

    async def _execute_research_workflow(self, query: str) -> AgentResponse:
        """Execute research workflow — both agents share all tools.
        """
        try:
            all_tools = self._build_all_tools()

            researcher = AssistantAgent(
                name="Researcher",
                system_message=(
                    "You are a research agent with access to internet search, stock prices, weather, and file saving. "
                    "Use search_internet to find real-time information, fetch_url to read full articles, "
                    "get_stock for financial data, get_city_weather for weather data. "
                    "Always use tools to gather real data before answering. Cite your sources. Keep responses concise."
                ),
                model_client=self.model_client,
                tools=all_tools,
            )

            analyst = AssistantAgent(
                name="Analyst",
                system_message=(
                    "You are an analyst with access to the same tools as the Researcher. "
                    "Review the research findings, use tools to verify or enrich data if needed, "
                    "then provide a structured analysis with key takeaways. "
                    "You can also use save_file to persist the final report. Keep responses concise."
                ),
                model_client=self.model_client,
                tools=all_tools,
            )

            termination = MaxMessageTermination(max_messages=4)

            team = RoundRobinGroupChat(
                participants=[researcher, analyst],
                termination_condition=termination
            )

            # Single clean user-turn task
            task_message = f"Research this topic using available tools: {query}"
            stream = team.run_stream(task=task_message)

            steps = []
            final_result = ""
            tools_used = set()

            async for message in stream:
                if hasattr(message, 'content'):
                    content_str = str(message.content)
                    steps.append(f"{message.source}: {content_str[:100]}...")
                    final_result = content_str
                # Track tool calls if available
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        tools_used.add(tc.name if hasattr(tc, 'name') else str(tc))

            return AgentResponse(
                answer=final_result,
                steps=[{"source": "autogen", "content": s} for s in steps],
                tools_used=list(tools_used) or [t.__name__ for t in all_tools],
                final_step=True
            )
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            # Fallback to debate workflow if research fails
            logger.info("Falling back to debate workflow")
            return await self._execute_debate_workflow(query)
