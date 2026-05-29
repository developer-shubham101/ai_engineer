"""Research workflow: 6-agent pipeline — pure async, no AutoGen."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from ....interfaces import AgentResponse

logger = logging.getLogger(__name__)

_AGENTS = [
    ("Planner",      "Break research queries into structured tasks."),
    ("Researcher",   "Gather factual evidence with citations only."),
    ("Verifier",     "Verify sources, remove duplicates, check consistency."),
    ("Analyst",      "Synthesize verified findings into insights."),
    ("Evaluator",    "Critique analysis for hallucinations, gaps, and weak evidence."),
    ("ReportWriter", (
        "Convert final analysis into a professional research report with sections: "
        "Key Findings, Evidence, Risks, Conclusion."
    )),
]


async def execute_research_workflow(
    llm_fn: Callable[[str, str], str],
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """Six-agent research pipeline: Plan → Research → Verify → Analyse → Evaluate → Report."""
    logger.debug("[custom/research] START query_len=%d max_steps=%d", len(query), max_steps)

    steps: List[Dict[str, Any]] = []
    context = f"Research topic: {query}"
    final_result = ""

    for idx, (name, system) in enumerate(_AGENTS[:max_steps], start=1):
        task = f"{context}\n\nPrevious steps so far:\n" + "\n".join(
            f"[{s['agent']}]: {s['content'][:300]}" for s in steps
        ) if steps else context
        try:
            content = await llm_fn(system, task)
        except Exception as exc:
            logger.warning("[custom/research] agent=%s failed: %s", name, exc)
            content = f"[{name} unavailable: {exc}]"
        steps.append({"step": idx, "agent": name, "type": "reasoning", "content": content})
        final_result = content

    logger.debug("[custom/research] DONE steps=%d answer_len=%d", len(steps), len(final_result))
    return AgentResponse(
        answer=final_result,
        steps=steps,
        tools_used=[],
        final_step=True,
    )
