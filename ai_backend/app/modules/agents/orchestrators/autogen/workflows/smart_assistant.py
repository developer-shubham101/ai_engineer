"""Smart assistant workflow: ToolSelector → ToolExecutor → Summarizer."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ....interfaces import AgentResponse
from ...utils import extract_json_object, fallback_tool_plan, normalize_tool_plan
from ...utils import build_executor_steps, merge_steps, run_team
from ...utils import get_tool_registry, build_tool_catalog, execute_tool_calls

logger = logging.getLogger(__name__)


async def _select_tools(
    model_client: Any,
    query: str,
    available_tool_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Agent 1 — ToolSelector: decide which tools to call and with what args."""
    if not available_tool_names:
        return {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": [], "routing_source": "none"}, []

    catalog = build_tool_catalog(available_tool_names)
    selector = AssistantAgent(
        name="ToolSelector",
        system_message=(
            "You are a tool selector. Analyse the user query and decide which tools are needed "
            "with exact arguments. Return ONLY valid JSON — no prose, no markdown fences.\n"
            'JSON shape: {"intent":"SHORT_INTENT","confidence":0.9,'
            '"tool_calls":[{"name":"tool_name","args":{"arg":"value"}}]}\n'
            "Rules:\n"
            "- Use only tools from the catalog.\n"
            "- Prefer specific tools over web_search when a direct tool fits.\n"
            "- Use scrape_url only when a URL is in the query.\n"
            "- Never select save/report tools."
        ),
        model_client=model_client,
    )
    team = RoundRobinGroupChat(
        participants=[selector],
        termination_condition=MaxMessageTermination(max_messages=2),
    )
    task = (
        f"Tool catalog:\n{json.dumps(catalog, indent=2, default=str)}\n\n"
        f"User query: {query}\n\n"
        "Return only the JSON tool plan."
    )

    try:
        result, steps, _ = await run_team(team, task)
        parsed = extract_json_object(result)
        logger.debug("ToolSelector parsed plan: %s", parsed)
        if not parsed:
            raise ValueError(f"Selector did not return JSON: {result!r}")
        return normalize_tool_plan(parsed, query, available_tool_names), steps
    except Exception as exc:
        logger.warning("AutoGen selector failed; falling back: %s", exc, exc_info=True)
        return fallback_tool_plan(query, available_tool_names), []


async def execute_smart_assistant_workflow(
    model_client: Any,
    tool_cache: Dict[str, Any],
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """3-agent pipeline: ToolSelector → ToolExecutor (deterministic) → Summarizer."""
    logger.debug("[smart_assistant] START query_len=%d tools=%s max_steps=%d",
                 len(query), [t.__name__ for t in tools], max_steps)
    registry = get_tool_registry()
    available_tool_names = [
        name for name, func in registry.items()
        if (not tools or func in tools) and name != "save_research_report"
    ]

    # Agent 1: Tool Selector
    route_plan, selector_steps = await _select_tools(model_client, query, available_tool_names)
    intent = route_plan["intent"]
    confidence = route_plan["confidence"]
    tool_calls = route_plan["tool_calls"]
    selected_tool_names = [tc["name"] for tc in tool_calls]
    logger.debug("[smart_assistant] selector intent=%s confidence=%s tools=%s routing=%s",
                 intent, confidence, selected_tool_names, route_plan.get("routing_source"))

    if not tool_calls and "web_search" in available_tool_names:
        tool_calls = [{"name": "web_search", "args": {"query": query}}]
        selected_tool_names = ["web_search"]

    if not selector_steps:
        selector_steps = [{
            "step": 1, "agent": "ToolSelector", "type": "tool_routing",
            "content": json.dumps(
                {"intent": intent, "confidence": confidence,
                 "routing_source": route_plan.get("routing_source"), "tool_calls": tool_calls},
                default=str,
            ),
        }]

    # Agent 2: Tool Executor (deterministic)
    tool_results = await execute_tool_calls(tool_calls, tool_cache)
    logger.debug("[smart_assistant] executor got %d results", len(tool_results))
    executor_steps = build_executor_steps(tool_results)
    executor_tools_used = {r["tool"] for r in tool_results}
    executor_result = json.dumps(tool_results, indent=2, default=str)

    # Agent 3: Summarizer
    summarizer = AssistantAgent(
        name="Summarizer",
        system_message=(
            "You are the final assistant. Tool results are already provided — do not call any tools. "
            "Summarize the results clearly and concisely. "
            "When the answer contains multiple independent facts (e.g. weather + stock price), "
            "return plain text always in formatted way so user can read."
        ),
        model_client=model_client,
    )
    summarizer_team = RoundRobinGroupChat(
        participants=[summarizer],
        termination_condition=MaxMessageTermination(max_messages=max_steps),
    )
    summarizer_task = (
        f"User query: {query}\n"
        f"Detected intent: {intent}\n"
        f"Tools used: {json.dumps(selected_tool_names)}\n"
        f"Tool results:\n{executor_result}"
    )
    final_result, summary_steps, summary_tools_used = await run_team(summarizer_team, summarizer_task)
    logger.debug("[smart_assistant] DONE steps=%d tools_used=%s answer_len=%d",
                 len(selector_steps) + len(executor_steps) + len(summary_steps),
                 selected_tool_names, len(final_result))

    all_steps = merge_steps(selector_steps + executor_steps, summary_steps)
    tools_used = (executor_tools_used or set(selected_tool_names)) | summary_tools_used

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
