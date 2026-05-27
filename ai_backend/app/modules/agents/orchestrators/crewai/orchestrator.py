"""CrewAI orchestrator — implements IAgentOrchestrator using the official CrewAI library."""
from __future__ import annotations

import json
import logging
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse
from .interfaces import ICrewOrchestrator, CrewRequest, CrewResponse
from .travel_workflow import run_smart_travel_planner
from ....config.settings import settings

logger = logging.getLogger(__name__)

# Resolved once at import time — crew_config/ sits at project root
_CONFIG_PATH = Path(settings.PROJECT_ROOT) / "crew_config"


def _load_yaml(filename: str) -> Dict[str, Any]:
    try:
        with open(_CONFIG_PATH / filename, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("CrewAI: failed to load %s: %s", filename, e)
        return {}


class CrewAIOrchestrator(IAgentOrchestrator, ICrewOrchestrator):
    """CrewAI multi-agent orchestrator.

    Implements both IAgentOrchestrator (used by /api/agents/query) and
    ICrewOrchestrator so it can be used standalone if needed.

    Workflows: debate, research, smart_assistant, smart_travel_planner, prompt_evaluation
    """

    AVAILABLE_WORKFLOWS: List[str] = [
        "debate", "research", "smart_assistant", "smart_travel_planner", "prompt_evaluation",
    ]
    AVAILABLE_TOOLS: List[str] = []  # CrewAI uses YAML-configured agents, not the tool registry

    def __init__(self) -> None:
        self._agents_config = _load_yaml("agents.yaml")
        self._tasks_config = _load_yaml("tasks.yaml")
        self.llm = self._create_llm()
        logger.debug("[CrewAI] orchestrator ready | llm=%s | workflows=%s",
                     self.llm, self.AVAILABLE_WORKFLOWS)

    # ------------------------------------------------------------------
    # LLM factory
    # ------------------------------------------------------------------

    def _create_llm(self) -> Optional[LLM]:
        try:
            logger.info("[CrewAI] creating LLM | base_url=%s", settings.CREW_BASE_URL)
            llm = LLM(model="llama", base_url=settings.CREW_BASE_URL, temperature=0.7)
            logger.info("[CrewAI] LLM ready")
            return llm
        except Exception as e:
            logger.error("[CrewAI] LLM creation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Agent / Task builders
    # ------------------------------------------------------------------

    def _agent(self, key: str) -> Agent:
        if self.llm is None:
            raise RuntimeError("CrewAI LLM not initialised — check CREW_BASE_URL")
        cfg = self._agents_config.get(key, {})
        return Agent(
            role=cfg.get("role", key),
            goal=cfg.get("goal", f"Complete tasks as {key}"),
            backstory=cfg.get("backstory", f"You are a {key} agent."),
            verbose=cfg.get("verbose", True),
            llm=self.llm,
        )

    def _task(self, key: str, agent: Agent, topic: str) -> Task:
        cfg = self._tasks_config.get(key, {})
        return Task(
            description=cfg.get("description", f"Complete task for {topic}").format(topic=topic),
            expected_output=cfg.get("expected_output", "Task completion summary"),
            agent=agent,
        )

    def _run_crew(self, agents: List[Agent], tasks: List[Task]) -> str:
        crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
        return str(crew.kickoff())

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    async def process_request(
        self, request: AgentRequest, user: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Dispatch AgentRequest to the correct workflow, return AgentResponse."""
        workflow = request.workflow.lower()
        if workflow not in self.AVAILABLE_WORKFLOWS:
            return AgentResponse(
                answer=f"Unknown workflow '{workflow}'. Available: {self.AVAILABLE_WORKFLOWS}",
                steps=[], tools_used=[], final_step=True,
            )

        logger.debug("[CrewAI] dispatch workflow=%s max_steps=%d", workflow, request.max_steps)
        start = time.perf_counter()

        crew_req = CrewRequest(
            topic=request.question,
            workflow_type=workflow,
            max_iterations=request.max_steps,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
        )
        crew_resp = await self.execute_workflow(crew_req, user)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.debug("[CrewAI] done workflow=%s agents=%s duration_ms=%s answer_len=%d",
                     workflow, crew_resp.agents_used, duration_ms, len(crew_resp.result))

        steps = [
            {"step": i + 1, "agent": a, "type": "reasoning", "content": ""}
            for i, a in enumerate(crew_resp.agents_used)
        ]
        return AgentResponse(
            answer=crew_resp.result,
            steps=steps,
            tools_used=[],
            final_step=True,
            debug_info={
                "workflow": workflow,
                "agents_used": crew_resp.agents_used,
                "iterations": crew_resp.iterations,
                "execution_time_ms": crew_resp.execution_time_ms,
                **(crew_resp.debug_info or {}),
            },
        )

    def register_tool(self, tool: Any) -> None:
        pass  # tools configured via YAML agents

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return self.AVAILABLE_WORKFLOWS

    # ------------------------------------------------------------------
    # ICrewOrchestrator interface
    # ------------------------------------------------------------------

    async def execute_workflow(
        self, request: CrewRequest, user: Optional[Dict[str, Any]] = None
    ) -> CrewResponse:
        """Execute a named CrewAI workflow and return CrewResponse."""
        start = time.time()
        try:
            wf = request.workflow_type
            if wf == "debate":
                result, agents_used = self._debate(request.topic)
            elif wf == "research":
                result, agents_used = self._research(request.topic)
            elif wf == "smart_assistant":
                result, agents_used = self._smart_assistant(request.topic)
            elif wf == "prompt_evaluation":
                result, agents_used = self._prompt_evaluation(request.topic)
            elif wf == "smart_travel_planner":
                return await self._travel_planner(request, start)
            else:
                raise ValueError(f"Unknown workflow_type: {wf!r}")

            return CrewResponse(
                result=result,
                workflow_type=wf,
                agents_used=agents_used,
                iterations=len(agents_used),
                execution_time_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            logger.error("[CrewAI] workflow failed: %s", e, exc_info=True)
            return CrewResponse(
                result=f"Workflow failed: {e}",
                workflow_type=request.workflow_type,
                agents_used=[],
                iterations=0,
                execution_time_ms=int((time.time() - start) * 1000),
                debug_info={"error": str(e)},
            )

    # ------------------------------------------------------------------
    # Workflow implementations
    # ------------------------------------------------------------------

    def _debate(self, topic: str) -> Tuple[str, List[str]]:
        """Advocate → Critic → Moderator."""
        advocate = self._agent("debate_advocate")
        critic = self._agent("debate_critic")
        moderator = self._agent("debate_moderator")
        tasks = [
            self._task("debate_advocate_task", advocate, topic),
            self._task("debate_critic_task", critic, topic),
            self._task("debate_moderator_task", moderator, topic),
        ]
        try:
            return self._run_crew([advocate, critic, moderator], tasks), ["Advocate", "Critic", "Moderator"]
        except Exception as e:
            logger.error("[CrewAI] debate failed: %s", e)
            return f"Debate failed: {e}", []

    def _research(self, topic: str) -> Tuple[str, List[str]]:
        """Researcher → Analyst → Synthesizer."""
        researcher = self._agent("researcher")
        analyst = self._agent("analyst")
        synthesizer = self._agent("synthesizer")
        tasks = [
            self._task("research_task", researcher, topic),
            self._task("analysis_task", analyst, topic),
            self._task("synthesis_task", synthesizer, topic),
        ]
        try:
            return self._run_crew([researcher, analyst, synthesizer], tasks), ["Researcher", "Analyst", "Synthesizer"]
        except Exception as e:
            logger.error("[CrewAI] research failed: %s", e)
            return f"Research failed: {e}", []

    def _smart_assistant(self, topic: str) -> Tuple[str, List[str]]:
        """ToolSelector → Summarizer."""
        selector = self._agent("tool_selector")
        summarizer = self._agent("assistant_summarizer")
        tasks = [
            self._task("tool_selector_task", selector, topic),
            self._task("assistant_summarizer_task", summarizer, topic),
        ]
        try:
            return self._run_crew([selector, summarizer], tasks), ["ToolSelector", "Summarizer"]
        except Exception as e:
            logger.error("[CrewAI] smart_assistant failed: %s", e)
            return f"Smart assistant failed: {e}", []

    def _prompt_evaluation(self, topic: str) -> Tuple[str, List[str]]:
        """PromptParser → CriteriaJudge → Improver → EvalReporter."""
        parser = self._agent("prompt_parser")
        judge = self._agent("criteria_judge")
        improver = self._agent("prompt_improver")
        reporter = self._agent("eval_reporter")
        tasks = [
            self._task("prompt_parser_task", parser, topic),
            self._task("criteria_judge_task", judge, topic),
            self._task("prompt_improver_task", improver, topic),
            self._task("eval_reporter_task", reporter, topic),
        ]
        try:
            return self._run_crew([parser, judge, improver, reporter], tasks), \
                   ["PromptParser", "CriteriaJudge", "Improver", "EvalReporter"]
        except Exception as e:
            logger.error("[CrewAI] prompt_evaluation failed: %s", e)
            return f"Prompt evaluation failed: {e}", []

    async def _travel_planner(self, request: CrewRequest, start: float) -> CrewResponse:
        """Lightweight tool-driven travel planner (no LLM agents needed)."""
        try:
            plan, agents_used = await run_smart_travel_planner(request.topic)
            return CrewResponse(
                result=json.dumps(plan, ensure_ascii=False, indent=2),
                workflow_type="smart_travel_planner",
                agents_used=agents_used,
                iterations=len(agents_used),
                execution_time_ms=int((time.time() - start) * 1000),
                debug_info={
                    "intent": plan.get("intent"),
                    "tools_used": plan.get("_meta", {}).get("tools_used", []),
                },
            )
        except Exception as e:
            logger.error("[CrewAI] travel_planner failed: %s", e)
            return CrewResponse(
                result=f"Travel planner failed: {e}",
                workflow_type="smart_travel_planner",
                agents_used=[],
                iterations=0,
                execution_time_ms=int((time.time() - start) * 1000),
                debug_info={"error": str(e)},
            )
