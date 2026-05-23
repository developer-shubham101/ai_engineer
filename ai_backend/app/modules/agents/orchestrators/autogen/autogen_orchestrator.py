"""AutoGen-based agent orchestrator for multi-agent conversations using AutoGen v0.4."""

import logging
from typing import Dict, Any, Optional, List, Callable

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

# Registry of all available tool builder functions — names match agent_runner.REGISTRY
_TOOL_BUILDERS: Dict[str, Callable] = {}


def _register_tool_builders() -> Dict[str, Callable]:
    """Lazily build the tool-name → callable map.

    Names are kept in sync with agent_runner.REGISTRY so /tools and
    /tools/{tool_name}/test work for both custom and AutoGen orchestrators.
    """
    if _TOOL_BUILDERS:
        return _TOOL_BUILDERS

    from ...function_tools.tool_web_search import web_search
    from ...function_tools.tool_web_scraper import scrape_url
    from ...function_tools.tool_stock import get_stock_price
    from ...function_tools.tool_weather import get_weather
    from ...function_tools.tool_file import save_text_file

    def web_search_tool(query: str) -> str:
        """Search the internet for real-time information on any topic."""
        result = web_search(query, max_results=5)
        return result["formatted"] if result.get("status") == "success" else f"Search failed: {result.get('error')}"

    def scrape_url_tool(url: str) -> str:
        """Fetch and extract full text content from a URL."""
        result = scrape_url(url)
        return result["content"] if result.get("status") == "success" else f"Fetch failed: {result.get('error')}"

    def get_stock_price_tool(symbol: str) -> str:
        """Get the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
        result = get_stock_price(symbol)
        return f"{result['symbol']}: ${result['price']}" if result.get("status") == "success" else f"Stock lookup failed: {result.get('error')}"

    def get_weather_tool(city: str) -> str:
        """Get current weather conditions for a city."""
        result = get_weather(city)
        if result.get("status") in ("success", "demo_data"):
            return f"{result['city']}: {result['temperature']}, {result['description']}, humidity {result['humidity']}"
        return f"Weather lookup failed: {result.get('error')}"

    def save_text_file_tool(filename: str, content: str) -> str:
        """Save text content to a file in user_uploaded_files/."""
        result = save_text_file(filename, content)
        return f"Saved '{result['filename']}' ({result['size']} chars)" if result.get("status") == "success" else f"Save failed: {result.get('error')}"

    _TOOL_BUILDERS.update({
        "web_search": web_search_tool,
        "scrape_url": scrape_url_tool,
        "get_stock_price": get_stock_price_tool,
        "get_weather": get_weather_tool,
        "save_text_file": save_text_file_tool,
    })
    return _TOOL_BUILDERS


class AutoGenOrchestrator(IAgentOrchestrator):
    """AutoGen-based multi-agent orchestrator using v0.4 API.

    Workflow and tools are fully controlled by the API caller via AgentRequest:
      - request.workflow  → which workflow to run (debate, research, ...)
      - request.tools     → which tools to inject (empty = all available)
    """

    # Names match agent_runner.REGISTRY for unified /tools discovery
    AVAILABLE_TOOLS = ["web_search", "scrape_url", "get_stock_price", "get_weather", "save_text_file"]

    # Map workflow name → handler method name
    WORKFLOW_REGISTRY = {
        "debate": "_execute_debate_workflow",
        "research": "_execute_research_workflow",
    }

    def __init__(self, model_client):
        self.model_client = model_client

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Dispatch to the workflow specified in request.workflow."""
        workflow = request.workflow.lower()
        handler_name = self.WORKFLOW_REGISTRY.get(workflow)

        if not handler_name:
            available = list(self.WORKFLOW_REGISTRY.keys())
            return AgentResponse(
                answer=f"Unknown workflow '{workflow}'. Available: {available}",
                steps=[], tools_used=[], final_step=True
            )

        tools = self._resolve_tools(request.tools)
        handler = getattr(self, handler_name)

        try:
            return await handler(request.question, tools, request.max_steps)
        except Exception as e:
            logger.error("AutoGen workflow '%s' failed: %s", workflow, e, exc_info=True)
            return AgentResponse(answer=f"Workflow failed: {e}", steps=[], tools_used=[], final_step=True)

    def register_tool(self, tool: Any) -> None:
        pass  # Tool registration handled via request.tools

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return list(self.WORKFLOW_REGISTRY.keys())

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def _resolve_tools(self, requested: List[str]) -> List[Callable]:
        """Return tool callables for the requested names (empty = all)."""
        registry = _register_tool_builders()
        names = requested if requested else list(registry.keys())
        tools = []
        for name in names:
            if name in registry:
                tools.append(registry[name])
            else:
                logger.warning("AutoGen: unknown tool '%s' requested, skipping", name)
        return tools

    # ------------------------------------------------------------------
    # Shared stream runner
    # ------------------------------------------------------------------

    async def _run_team(self, team: RoundRobinGroupChat, task: str) -> tuple[str, List[Dict[str, Any]], set]:
        steps, tools_used, final_result = [], set(), ""
        step_index = 0
        async for message in team.run_stream(task=task):
            if hasattr(message, "content"):
                step_index += 1
                content_str = str(message.content)
                step: Dict[str, Any] = {
                    "step": step_index,
                    "agent": getattr(message, "source", "unknown"),
                    "content": content_str,
                    "type": "tool_call" if (hasattr(message, "tool_calls") and message.tool_calls) else "reasoning",
                }
                if hasattr(message, "tool_calls") and message.tool_calls:
                    step["tools_called"] = [
                        tc.name if hasattr(tc, "name") else str(tc)
                        for tc in message.tool_calls
                    ]
                    for tc in message.tool_calls:
                        tools_used.add(tc.name if hasattr(tc, "name") else str(tc))
                steps.append(step)
                final_result = content_str
        return final_result, steps, tools_used

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    async def _execute_debate_workflow(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        """Three-agent debate: Advocate vs Critic, moderated by Moderator."""
        advocate = AssistantAgent(
            name="Advocate",
            system_message="You argue FOR the given topic with strong supporting evidence. Be concise.",
            model_client=self.model_client,
            tools=tools or None,
        )
        critic = AssistantAgent(
            name="Critic",
            system_message="You argue AGAINST the given topic with counterarguments. Be concise.",
            model_client=self.model_client,
            tools=tools or None,
        )
        moderator = AssistantAgent(
            name="Moderator",
            system_message="Moderate the debate and provide a final balanced summary. Be concise.",
            model_client=self.model_client,
        )

        team = RoundRobinGroupChat(
            participants=[advocate, critic, moderator],
            termination_condition=MaxMessageTermination(max_messages=max_steps)
        )
        final_result, steps, tools_used = await self._run_team(team, f"Debate topic: {query}")

        return AgentResponse(
            answer=final_result,
            steps=steps,
            tools_used=list(tools_used) or [t.__name__ for t in tools],
            final_step=True
        )

    async def _execute_research_workflow(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        """Two-agent research: Researcher gathers data, Analyst synthesizes."""
        researcher = AssistantAgent(
            name="Researcher",
            system_message=(
                "You are a research agent. Use available tools to gather real-time data. "
                "Always cite sources. Be concise."
            ),
            model_client=self.model_client,
            tools=tools or None,
        )
        analyst = AssistantAgent(
            name="Analyst",
            system_message=(
                "You are an analyst. Review research findings, enrich with tools if needed, "
                "then provide a structured analysis with key takeaways. Be concise."
            ),
            model_client=self.model_client,
            tools=tools or None,
        )

        team = RoundRobinGroupChat(
            participants=[researcher, analyst],
            termination_condition=MaxMessageTermination(max_messages=max_steps)
        )
        final_result, steps, tools_used = await self._run_team(team, f"Research this topic: {query}")

        return AgentResponse(
            answer=final_result,
            steps=steps,
            tools_used=list(tools_used) or [t.__name__ for t in tools],
            final_step=True
        )
