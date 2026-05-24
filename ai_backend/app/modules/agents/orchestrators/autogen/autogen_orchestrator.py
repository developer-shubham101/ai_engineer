"""AutoGen-based multi-agent orchestrator using AutoGen v0.4."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse
from .tool_registry import get_tool_registry
from .tool_utils import resolve_tools
from .workflows import (
    execute_debate_workflow,
    execute_research_workflow,
    execute_smart_assistant_workflow,
    execute_smart_travel_planner_workflow,
)

logger = logging.getLogger(__name__)


class AutoGenOrchestrator(IAgentOrchestrator):
    """AutoGen multi-agent orchestrator.

    Workflow and tools are fully controlled by the API caller via AgentRequest:
      - request.workflow  → which workflow to run (debate, research, ...)
      - request.tools     → which tools to inject (empty = all available)
    """

    # Names match agent_runner.REGISTRY for unified /tools discovery
    AVAILABLE_TOOLS: List[str] = [
        "web_search", "scrape_url",
        "get_stock_price", "get_stock_history", "generate_stock_chart",
        "get_crypto_price", "generate_chart", "get_weather",
        "save_research_report",
        "search_flights", "search_hotels", "estimate_trip_budget",
        "search_places", "search_restaurants", "generate_itinerary",
        "get_local_transport_info", "get_distance_between_places",
        "generate_trip_summary", "get_currency_exchange", "get_geo_distance",
    ]

    WORKFLOW_REGISTRY: Dict[str, Callable] = {}  # populated in __init__

    def __init__(self, model_client: Any) -> None:
        self.model_client = model_client
        self._tool_cache: Dict[str, Any] = {}
        self.WORKFLOW_REGISTRY = {
            "debate": self._run_debate,
            "research": self._run_research,
            "smart_assistant": self._run_smart_assistant,
            "smart_travel_planner": self._run_smart_travel_planner,
        }

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    async def process_request(
        self, request: AgentRequest, user: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Dispatch to the workflow specified in request.workflow."""
        workflow = request.workflow.lower()
        handler = self.WORKFLOW_REGISTRY.get(workflow)

        if not handler:
            return AgentResponse(
                answer=f"Unknown workflow '{workflow}'. Available: {list(self.WORKFLOW_REGISTRY)}",
                steps=[], tools_used=[], final_step=True,
            )

        tools = resolve_tools(request.tools)
        logger.debug(
            "[AutoGen] dispatch workflow=%s tools=%s max_steps=%s",
            workflow, [t.__name__ for t in tools], request.max_steps,
        )
        try:
            response = await handler(request.question, tools, request.max_steps)
            logger.debug(
                "[AutoGen] workflow=%s done steps=%d tools_used=%s answer_len=%d",
                workflow, len(response.steps), response.tools_used,
                len(response.answer or ""),
            )
            return response
        except Exception as e:
            logger.error("AutoGen workflow '%s' failed: %s", workflow, e, exc_info=True)
            return AgentResponse(answer=f"Workflow failed: {e}", steps=[], tools_used=[], final_step=True)

    def register_tool(self, tool: Any) -> None:
        pass  # Tool registration handled via request.tools

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return list(self.WORKFLOW_REGISTRY)

    # ------------------------------------------------------------------
    # Workflow dispatchers (thin wrappers that inject shared state)
    # ------------------------------------------------------------------

    async def _run_debate(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_debate_workflow(self.model_client, query, tools, max_steps)

    async def _run_research(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_research_workflow(self.model_client, query, tools, max_steps)

    async def _run_smart_assistant(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_smart_assistant_workflow(self.model_client, self._tool_cache, query, tools, max_steps)

    async def _run_smart_travel_planner(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_smart_travel_planner_workflow(self.model_client, self._tool_cache, query, tools, max_steps)
