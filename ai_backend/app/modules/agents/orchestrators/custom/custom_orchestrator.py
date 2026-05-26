"""Custom multi-agent orchestrator — same 4 workflows as AutoGen, no AutoGen dependency."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse
from ..utils import get_tool_registry, resolve_tools
from .workflows import (
    execute_debate_workflow,
    execute_research_workflow,
    execute_smart_assistant_workflow,
    execute_smart_travel_planner_workflow,
)

logger = logging.getLogger(__name__)


class CustomOrchestrator(IAgentOrchestrator):
    """Pure-Python multi-agent orchestrator.

    Mirrors AutoGenOrchestrator workflow-for-workflow:
      debate               → Advocate / Critic / Moderator  (sequential LLM calls)
      research             → 6-agent pipeline               (sequential LLM calls)
      smart_assistant      → ToolSelector → ToolExecutor → Summarizer
      smart_travel_planner → TravelToolSelector → ToolExecutor → TravelPlanner

    The only structural difference from AutoGenOrchestrator:
      AutoGen uses AssistantAgent + RoundRobinGroupChat.
      Custom  uses a plain async llm_fn(system, user) → str.
    """

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

    AVAILABLE_WORKFLOWS: List[str] = [
        "debate", "research", "smart_assistant", "smart_travel_planner",
    ]

    def __init__(self, llm_fn: Optional[Callable[[str, str], Any]] = None) -> None:
        """
        Args:
            llm_fn: async callable(system_prompt, user_prompt) -> str.
                    Falls back to a no-op echo if not provided.
        """
        self._llm_fn = llm_fn or self._echo_llm
        self._tool_cache: Dict[str, Any] = {}
        self.WORKFLOW_REGISTRY: Dict[str, Callable] = {
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
        logger.debug("[Custom] dispatch workflow=%s tools=%s max_steps=%s",
                     workflow, [t.__name__ for t in tools], request.max_steps)
        try:
            response = await handler(request.question, tools, request.max_steps)
            logger.debug("[Custom] workflow=%s done steps=%d tools_used=%s answer_len=%d",
                         workflow, len(response.steps), response.tools_used,
                         len(response.answer or ""))
            return response
        except Exception as e:
            logger.error("Custom workflow '%s' failed: %s", workflow, e, exc_info=True)
            return AgentResponse(answer=f"Workflow failed: {e}", steps=[], tools_used=[], final_step=True)

    def register_tool(self, tool: Any) -> None:
        pass  # Tools resolved from shared registry via request.tools

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return self.AVAILABLE_WORKFLOWS

    # ------------------------------------------------------------------
    # Workflow dispatchers
    # ------------------------------------------------------------------

    async def _run_debate(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_debate_workflow(self._llm_fn, query, tools, max_steps)

    async def _run_research(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_research_workflow(self._llm_fn, query, tools, max_steps)

    async def _run_smart_assistant(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_smart_assistant_workflow(self._llm_fn, self._tool_cache, query, tools, max_steps)

    async def _run_smart_travel_planner(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        return await execute_smart_travel_planner_workflow(self._llm_fn, self._tool_cache, query, tools, max_steps)

    # ------------------------------------------------------------------
    # Fallback LLM
    # ------------------------------------------------------------------

    @staticmethod
    async def _echo_llm(system: str, user: str) -> str:
        """No-op LLM used when no llm_fn is injected (returns user prompt as-is)."""
        await asyncio.sleep(0)
        return user
