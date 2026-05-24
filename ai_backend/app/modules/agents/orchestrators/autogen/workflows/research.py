"""Research workflow: 6-agent pipeline Plan→Research→Verify→Analyse→Evaluate→Report."""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ....interfaces import AgentResponse
from ..step_utils import run_team

logger = logging.getLogger(__name__)


def _research_tools(tools: List[Callable]) -> List[Callable]:
    return [t for t in tools if t.__name__ != "save_text_file_tool"]


def _save_tools(tools: List[Callable]) -> List[Callable]:
    return [t for t in tools if t.__name__ == "save_research_report_tool"]


async def execute_research_workflow(
    model_client: Any,
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """Six-agent research pipeline: Plan → Research → Verify → Analyse → Evaluate → Report."""
    logger.debug("[research] START query_len=%d tools=%s max_steps=%d",
                 len(query), [t.__name__ for t in tools], max_steps)
    planner = AssistantAgent(
        name="Planner",
        system_message="Break research queries into structured tasks.",
        model_client=model_client,
    )
    researcher = AssistantAgent(
        name="Researcher",
        system_message="Gather factual evidence with citations only.",
        model_client=model_client,
        tools=_research_tools(tools) or None,
    )
    verifier = AssistantAgent(
        name="Verifier",
        system_message="Verify sources, remove duplicates, check consistency.",
        model_client=model_client,
    )
    analyst = AssistantAgent(
        name="Analyst",
        system_message="Synthesize verified findings into insights.",
        model_client=model_client,
    )
    evaluator = AssistantAgent(
        name="Evaluator",
        system_message="Critique analysis for hallucinations, gaps, and weak evidence.",
        model_client=model_client,
    )
    report_writer = AssistantAgent(
        name="ReportWriter",
        system_message=(
            "Convert final analysis into a professional research report.\n"
            "Call save_research_report with:\n"
            "  title    = concise report title\n"
            "  query    = the original research question\n"
            "  summary  = 1-3 sentence executive summary\n"
            "  markdown = full report body in markdown (Key Findings, Evidence, Risks, Conclusion)\n"
            "  metadata = JSON string with tags and topic, e.g. '{\"topic\": \"AI\", \"tags\": [\"research\"]}'\n"
            "  sources  = newline-separated URLs or citations from Researcher"
        ),
        model_client=model_client,
        tools=_save_tools(tools) or None,
    )

    team = RoundRobinGroupChat(
        participants=[planner, researcher, verifier, analyst, evaluator, report_writer],
        termination_condition=MaxMessageTermination(max_messages=max_steps),
    )
    task = (
        f"Research this topic thoroughly:\n\n{query}\n\n"
        "Final step: Save the final report using save_research_report tool."
    )
    final_result, steps, tools_used = await run_team(team, task)
    logger.debug("[research] DONE steps=%d tools_used=%s answer_len=%d",
                 len(steps), tools_used, len(final_result))

    return AgentResponse(
        answer=final_result,
        steps=steps,
        tools_used=list(tools_used),
        final_step=True,
    )
