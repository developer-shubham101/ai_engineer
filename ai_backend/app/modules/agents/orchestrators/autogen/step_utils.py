"""Shared step-building and team-running utilities for AutoGen workflows."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set, Tuple

from autogen_agentchat.teams import RoundRobinGroupChat

logger = logging.getLogger(__name__)


async def run_team(
    team: RoundRobinGroupChat,
    task: str,
) -> Tuple[str, List[Dict[str, Any]], Set[str]]:
    """Stream a RoundRobinGroupChat and collect steps + tools used.

    Returns:
        (final_result, steps, tools_used)
    """
    steps: List[Dict[str, Any]] = []
    tools_used: Set[str] = set()
    last_non_empty = ""
    step_index = 0

    logger.debug("[run_team] starting task (first 120 chars): %s", task[:120])

    async for message in team.run_stream(task=task):
        if not hasattr(message, "content") or message.content is None:
            continue
        content_str = str(message.content).strip()
        if not content_str:
            continue

        step_index += 1
        agent = getattr(message, "source", "unknown")
        step: Dict[str, Any] = {
            "step": step_index,
            "agent": agent,
            "content": content_str,
            "type": "tool_call" if (hasattr(message, "tool_calls") and message.tool_calls) else "reasoning",
        }

        if hasattr(message, "tool_calls") and message.tool_calls:
            step["tools_called"] = []
            for tc in message.tool_calls:
                tool_name = tc.name if hasattr(tc, "name") else str(tc)
                step["tools_called"].append(tool_name)
                tools_used.add(tool_name)
            logger.debug("[run_team] step=%d agent=%s tool_calls=%s", step_index, agent, step["tools_called"])
        else:
            logger.debug("[run_team] step=%d agent=%s content_len=%d", step_index, agent, len(content_str))

        steps.append(step)
        last_non_empty = content_str

    logger.debug("[run_team] finished total_steps=%d tools_used=%s answer_len=%d",
                 len(steps), tools_used, len(last_non_empty))
    return last_non_empty, steps, tools_used


def build_executor_steps(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw tool result envelopes into step dicts for the response."""
    return [
        {
            "step": idx,
            "agent": "ToolExecutor",
            "type": "tool_execution",
            "tool": result["tool"],
            "args": result["args"],
            "content": json.dumps(result["result"], indent=2, default=str),
            "duration_ms": result.get("duration_ms"),
            "cached": result.get("cached"),
        }
        for idx, result in enumerate(tool_results, start=1)
    ]


def merge_steps(
    pre_steps: List[Dict[str, Any]],
    post_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Renumber pre_steps from 1 and offset post_steps sequentially after them."""
    for i, step in enumerate(pre_steps, start=1):
        step["step"] = i
    offset = len(pre_steps)
    for step in post_steps:
        step["step"] = step.get("step", 0) + offset
    return pre_steps + post_steps
