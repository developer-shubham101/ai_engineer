"""Smart assistant workflow: ToolSelector → ToolExecutor → Summarizer — pure async, no AutoGen."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Tuple

from .....interfaces import AgentResponse
from ...utils import (
    build_tool_catalog, build_executor_steps, execute_tool_calls,
    extract_json_object, fallback_tool_plan, get_tool_registry,
    merge_steps, normalize_tool_plan,
)

logger = logging.getLogger(__name__)


async def _select_tools(
    llm_fn: Callable[[str, str], str],
    query: str,
    available_tool_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Agent 1 — ToolSelector: pick tools via a single LLM call."""
    if not available_tool_names:
        return {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": [], "routing_source": "none"}, []

    catalog = build_tool_catalog(available_tool_names)
    system = (
        "You are a tool selector. Analyse the user query and decide which tools are needed "
        "with exact arguments. Return ONLY valid JSON — no prose, no markdown fences.\n"
        'JSON shape: {"intent":"SHORT_INTENT","confidence":0.9,'
        '"tool_calls":[{"name":"tool_name","args":{"arg":"value"}}]}\n'
        "Rules:\n"
        "- Use only tools from the catalog.\n"
        "- Prefer specific tools over web_search when a direct tool fits.\n"
        "- Use scrape_url only when a URL is in the query.\n"
        "- Never select save/report tools."
    )
    task = (
        f"Tool catalog:\n{json.dumps(catalog, indent=2, default=str)}\n\n"
        f"User query: {query}\n\n"
        "Return only the JSON tool plan."
    )
    try:
        result = await llm_fn(system, task)
        parsed = extract_json_object(result)
        if not parsed:
            raise ValueError(f"ToolSelector did not return JSON: {result!r}")
        plan = normalize_tool_plan(parsed, query, available_tool_names)
        steps = [{"step": 1, "agent": "ToolSelector", "type": "tool_routing",
                  "content": json.dumps(plan, default=str)}]
        return plan, steps
    except Exception as exc:
        logger.warning("[custom/smart_assistant] selector failed, fallback: %s", exc)
        return fallback_tool_plan(query, available_tool_names), []


async def execute_smart_assistant_workflow(
    llm_fn: Callable[[str, str], str],
    tool_cache: Dict[str, Any],
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """3-agent pipeline: ToolSelector → ToolExecutor (deterministic) → Summarizer."""
    logger.debug("[custom/smart_assistant] START query_len=%d max_steps=%d", len(query), max_steps)
    registry = get_tool_registry()
    available_tool_names = [
        name for name, func in registry.items()
        if (not tools or func in tools) and name != "save_research_report"
    ]

    # Agent 1: ToolSelector
    route_plan, selector_steps = await _select_tools(llm_fn, query, available_tool_names)
    intent = route_plan["intent"]
    confidence = route_plan["confidence"]
    tool_calls = route_plan["tool_calls"]
    selected_tool_names = [tc["name"] for tc in tool_calls]

    if not tool_calls and "web_search" in available_tool_names:
        tool_calls = [{"name": "web_search", "args": {"query": query}}]
        selected_tool_names = ["web_search"]

    if not selector_steps:
        selector_steps = [{"step": 1, "agent": "ToolSelector", "type": "tool_routing",
                           "content": json.dumps({"intent": intent, "confidence": confidence,
                                                  "routing_source": route_plan.get("routing_source"),
                                                  "tool_calls": tool_calls}, default=str)}]

    # Agent 2: ToolExecutor (deterministic)
    tool_results = await execute_tool_calls(tool_calls, tool_cache)
    executor_steps = build_executor_steps(tool_results)
    executor_tools_used = {r["tool"] for r in tool_results}
    executor_result = json.dumps(tool_results, indent=2, default=str)

    # Agent 3: Summarizer
    system = (
        "You are the final assistant. Tool results are already provided — do not call any tools. "
        "Summarize the results clearly and concisely. "
        "When the answer contains multiple independent facts (e.g. weather + stock price), "
        "return plain text always in formatted way so user can read."
    )
    task = (
        f"User query: {query}\n"
        f"Detected intent: {intent}\n"
        f"Tools used: {json.dumps(selected_tool_names)}\n"
        f"Tool results:\n{executor_result}"
    )
    try:
        final_result = await llm_fn(system, task)
    except Exception as exc:
        logger.warning("[custom/smart_assistant] summarizer failed: %s", exc)
        final_result = executor_result

    summary_steps = [{"step": 1, "agent": "Summarizer", "type": "reasoning", "content": final_result}]

    all_steps = merge_steps(selector_steps + executor_steps, summary_steps)
    tools_used = (executor_tools_used or set(selected_tool_names))

    logger.debug("[custom/smart_assistant] DONE steps=%d tools_used=%s answer_len=%d",
                 len(all_steps), list(tools_used), len(final_result))
    return AgentResponse(
        answer=final_result,
        steps=all_steps,
        tools_used=list(tools_used),
        final_step=True,
        debug_info={
            "intent": intent,
            "confidence": confidence,
            "selected_tools": selected_tool_names,
            "routing_source": route_plan.get("routing_source"),
            "tool_calls": tool_calls,
        },
    )
