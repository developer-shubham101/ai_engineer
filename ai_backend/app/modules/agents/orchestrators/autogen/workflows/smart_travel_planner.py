"""Smart travel planner workflow: TravelToolSelector → ToolExecutor → TravelPlanner."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ....interfaces import AgentResponse
from ..json_utils import extract_json_object
from ..plan_normalizer import TRAVEL_TOOL_NAMES, fallback_travel_plan, normalize_travel_tool_plan
from ..step_utils import build_executor_steps, merge_steps, run_team
from ..tool_registry import get_tool_registry
from ..tool_utils import build_tool_catalog, execute_tool_calls

logger = logging.getLogger(__name__)


async def _select_travel_tools(
    model_client: Any,
    query: str,
    available_tool_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Agent 1 — TravelToolSelector: extract entities and select travel tools."""
    if not available_tool_names:
        return {
            "intent": "GENERAL_TRAVEL_QUERY", "confidence": 0.0,
            "entities": {}, "tool_calls": [], "routing_source": "none",
        }, []

    catalog = build_tool_catalog(available_tool_names)
    selector = AssistantAgent(
        name="TravelToolSelector",
        system_message=(
            "You are an AI travel routing agent.\n\n"
            "Your job:\n"
            "- Understand the user's travel request\n"
            "- Extract travel entities\n"
            "- Select the minimum required tools\n"
            "- Generate COMPLETE tool arguments\n\n"
            "Return ONLY valid JSON.\n"
            "No markdown.\n"
            "No explanation.\n\n"
            "JSON FORMAT:\n"
            "{\n"
            '  "intent": "TRAVEL_INTENT",\n'
            '  "confidence": 0.95,\n'
            '  "entities": {\n'
            '      "destination": "Goa",\n'
            '      "source": "Delhi",\n'
            '      "days": 3,\n'
            '      "budget": 25000,\n'
            '      "budget_currency": "INR",\n'
            '      "travelers": 2,\n'
            '      "preferences": ["beach","nightlife"]\n'
            "  },\n"
            '  "tool_calls": [\n'
            "      {\n"
            '          "name": "search_hotels",\n'
            '          "args": {\n'
            '              "destination": "Goa",\n'
            '              "budget": "25000",\n'
            '              "days": "3"\n'
            "          }\n"
            "      }\n"
            "  ]\n"
            "}\n\n"
            "RULES:\n"
            "- Use ONLY tools from catalog.\n"
            "- Do NOT use web_search.\n"
            "- Do NOT use scrape_url.\n"
            "- Generate COMPLETE args for every tool.\n"
            "- Avoid unnecessary tools.\n"
            "- If budget currency is non-INR and conversion is useful, use get_currency_exchange.\n"
            "- If source city is missing, omit tools needing origin unless required.\n"
            "- Use smart reasoning instead of keyword matching."
        ),
        model_client=model_client,
    )
    team = RoundRobinGroupChat(
        participants=[selector],
        termination_condition=MaxMessageTermination(max_messages=2),
    )
    task = (
        f"Available tools:\n{json.dumps(catalog, indent=2, default=str)}\n\n"
        f"User query:\n{query}\n\n"
        "Return ONLY JSON."
    )

    try:
        result, steps, _ = await run_team(team, task)
        parsed = extract_json_object(result)
        if not parsed:
            raise ValueError(f"Travel selector did not return valid JSON: {result!r}")
        return normalize_travel_tool_plan(parsed, query, available_tool_names), steps
    except Exception as exc:
        logger.warning("Travel selector failed, fallback triggered: %s", exc, exc_info=True)
        return fallback_travel_plan(query, available_tool_names), []


async def execute_smart_travel_planner_workflow(
    model_client: Any,
    tool_cache: Dict[str, Any],
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """3-agent pipeline: TravelToolSelector → ToolExecutor (deterministic) → TravelPlanner."""
    logger.debug("[travel_planner] START query_len=%d tools=%s max_steps=%d",
                 len(query), [t.__name__ for t in tools], max_steps)
    registry = get_tool_registry()
    available_tool_names = [
        name for name, func in registry.items()
        if (not tools or func in tools) and name in TRAVEL_TOOL_NAMES
    ]

    # Agent 1: Travel Tool Selector
    route_plan, selector_steps = await _select_travel_tools(model_client, query, available_tool_names)
    intent = route_plan["intent"]
    confidence = route_plan["confidence"]
    entities = route_plan.get("entities", {})
    tool_calls = route_plan["tool_calls"]
    selected_tool_names = [tc["name"] for tc in tool_calls]
    logger.debug("[travel_planner] selector intent=%s confidence=%s dest=%s tools=%s routing=%s",
                 intent, confidence, entities.get("destination"),
                 selected_tool_names, route_plan.get("routing_source"))

    if not selector_steps:
        selector_steps = [{
            "step": 1, "agent": "TravelToolSelector", "type": "tool_routing",
            "content": json.dumps(
                {"intent": intent, "confidence": confidence, "entities": entities,
                 "routing_source": route_plan.get("routing_source"), "tool_calls": tool_calls},
                default=str,
            ),
        }]

    logger.info(
        "TRAVEL_PLANNER: intent=%s | dest=%s | days=%s | budget=%s | tools=%s",
        intent, entities.get("destination"), entities.get("days"),
        entities.get("budget"), selected_tool_names,
    )

    # Agent 2: Tool Executor (deterministic)
    tool_results = await execute_tool_calls(tool_calls, tool_cache)
    logger.debug("[travel_planner] executor got %d results", len(tool_results))
    if not tool_results:
        return AgentResponse(
            answer="Unable to gather travel information right now.",
            steps=selector_steps, tools_used=[], final_step=True,
            debug_info={"intent": intent, "confidence": confidence,
                        "routing_source": route_plan.get("routing_source")},
        )

    executor_steps = build_executor_steps(tool_results)
    executor_tools_used = {r["tool"] for r in tool_results}
    executor_result = json.dumps(tool_results, indent=2, default=str)

    # Agent 3: Travel Planner
    preferences = entities.get("preferences") or []
    if not isinstance(preferences, list):
        preferences = [str(preferences)]
    preferences_str = ", ".join(preferences) or "general travel"

    planner = AssistantAgent(
        name="TravelPlanner",
        system_message=(
            "You are an expert AI travel planner.\n\n"
            "Tool results are already provided.\n"
            "Never call tools.\n\n"
            "Your job:\n"
            "- Build a clean travel plan\n"
            "- Combine all tool results intelligently\n"
            "- Remove duplicate information\n"
            "- Make recommendations when useful\n\n"
            "Always format response nicely using sections:\n"
            "- Overview\n- Budget\n- Hotels\n- Attractions\n- Weather\n- Transport\n- Tips\n\n"
            "If some information is unavailable, skip that section.\n"
            "Be concise but practical."
        ),
        model_client=model_client,
    )
    planner_team = RoundRobinGroupChat(
        participants=[planner],
        termination_condition=MaxMessageTermination(max_messages=max_steps),
    )
    planner_task = (
        f"User query: {query}\n"
        f"Detected intent: {intent}\n"
        f"Origin: {entities.get('source') or 'Not specified'} | Destination: {entities.get('destination')}\n"
        f"Duration: {entities.get('days')} days | Travelers: {entities.get('travelers')}\n"
        f"Preferences: {preferences_str}\n"
        f"Tools used: {json.dumps(selected_tool_names)}\n"
        f"Tool results:\n{executor_result}"
    )
    final_result, summary_steps, summary_tools_used = await run_team(planner_team, planner_task)
    logger.debug("[travel_planner] DONE steps=%d tools_used=%s answer_len=%d",
                 len(selector_steps) + len(executor_steps) + len(summary_steps),
                 list(executor_tools_used), len(final_result))

    all_steps = merge_steps(selector_steps + executor_steps, summary_steps)
    tools_used = (executor_tools_used or set(selected_tool_names)) | summary_tools_used

    return AgentResponse(
        answer=final_result,
        steps=all_steps,
        tools_used=list(tools_used),
        final_step=True,
        debug_info={
            "intent": intent, "confidence": confidence,
            "origin": entities.get("source"),
            "destination": entities.get("destination"),
            "days": entities.get("days"),
            "budget": entities.get("budget"),
            "budget_currency": entities.get("budget_currency"),
            "travelers": entities.get("travelers"),
            "preferences": entities.get("preferences", []),
            "selected_tools": selected_tool_names,
            "routing_source": route_plan.get("routing_source"),
            "tool_calls": tool_calls,
        },
    )
