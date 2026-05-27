"""CrewAI orchestrator adapter — wraps CrewOrchestrator, returns AgentResponse."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse
from ....crew_ai.orchestrator import CrewOrchestrator
from ....crew_ai.interfaces import CrewRequest

logger = logging.getLogger(__name__)


class CrewAIOrchestrator(IAgentOrchestrator):
    """Adapter that exposes CrewOrchestrator through the IAgentOrchestrator interface.

    Supports the same workflows as AutoGen/Custom:
      debate, research, smart_assistant, smart_travel_planner, prompt_evaluation

    Returns AgentResponse so /api/agents/query works identically for all orchestrator types.
    """

    AVAILABLE_WORKFLOWS: List[str] = [
        "debate", "research", "smart_assistant", "smart_travel_planner", "prompt_evaluation",
    ]

    # CrewAI does not use function tools from the registry
    AVAILABLE_TOOLS: List[str] = []

    def __init__(self) -> None:
        self._crew = CrewOrchestrator()
        logger.debug("[CrewAI] orchestrator initialised, llm=%s", self._crew.llm)

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    async def process_request(
        self, request: AgentRequest, user: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Translate AgentRequest → CrewRequest, run workflow, return AgentResponse."""
        workflow = request.workflow.lower()

        if workflow not in self.AVAILABLE_WORKFLOWS:
            return AgentResponse(
                answer=f"Unknown workflow '{workflow}'. Available: {self.AVAILABLE_WORKFLOWS}",
                steps=[], tools_used=[], final_step=True,
            )

        # Map smart_assistant → debate as the closest CrewAI equivalent when
        # the underlying crew doesn't have a dedicated smart_assistant workflow.
        crew_workflow = self._map_workflow(workflow)

        logger.debug("[CrewAI] dispatch workflow=%s (mapped=%s)", workflow, crew_workflow)
        start = time.perf_counter()

        crew_request = CrewRequest(
            topic=request.question,
            workflow_type=crew_workflow,
            max_iterations=request.max_steps,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
        )

        crew_response = await self._crew.execute_workflow(crew_request, user)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.debug(
            "[CrewAI] done workflow=%s agents=%s duration_ms=%s answer_len=%d",
            workflow, crew_response.agents_used, duration_ms, len(crew_response.result),
        )

        # Build steps from agents_used so the response shape matches AutoGen
        steps = [
            {"step": i + 1, "agent": agent, "type": "reasoning", "content": ""}
            for i, agent in enumerate(crew_response.agents_used)
        ]

        return AgentResponse(
            answer=crew_response.result,
            steps=steps,
            tools_used=[],
            final_step=True,
            debug_info={
                "workflow": workflow,
                "crew_workflow": crew_workflow,
                "agents_used": crew_response.agents_used,
                "iterations": crew_response.iterations,
                "execution_time_ms": crew_response.execution_time_ms,
                **(crew_response.debug_info or {}),
            },
        )

    def register_tool(self, tool: Any) -> None:
        pass  # CrewAI tools are configured inside CrewOrchestrator

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return self.AVAILABLE_WORKFLOWS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_workflow(workflow: str) -> str:
        """Pass workflow name directly — CrewOrchestrator handles all 5 workflows natively."""
        return workflow
