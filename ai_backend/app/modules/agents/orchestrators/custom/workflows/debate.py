"""Debate workflow: Advocate vs Critic vs Moderator — pure async, no AutoGen."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from interfaces import AgentResponse

logger = logging.getLogger(__name__)


async def execute_debate_workflow(
    llm_fn: Callable[[str, str], str],
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """Three-turn debate: Advocate → Critic → Moderator, each calling llm_fn."""
    logger.debug("[custom/debate] START query_len=%d max_steps=%d", len(query), max_steps)

    agents = [
        ("Advocate",  "You argue FOR the given topic with strong supporting evidence. Be concise."),
        ("Critic",    "You argue AGAINST the given topic with counterarguments. Be concise."),
        ("Moderator", "Moderate the debate and provide a final balanced summary. Be concise."),
    ]

    steps: List[Dict[str, Any]] = []
    final_result = ""

    for idx, (name, system) in enumerate(agents[:max_steps], start=1):
        try:
            content = await llm_fn(system, f"Debate topic: {query}")
        except Exception as exc:
            logger.warning("[custom/debate] agent=%s failed: %s", name, exc)
            content = f"[{name} unavailable: {exc}]"
        steps.append({"step": idx, "agent": name, "type": "reasoning", "content": content})
        final_result = content

    logger.debug("[custom/debate] DONE steps=%d answer_len=%d", len(steps), len(final_result))
    return AgentResponse(
        answer=final_result,
        steps=steps,
        tools_used=[],
        final_step=True,
    )
