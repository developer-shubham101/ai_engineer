"""Plan normalization: parse and validate LLM tool-selection outputs."""
from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from .tool_registry import get_tool_registry

logger = logging.getLogger(__name__)


def _parse_confidence(raw: Any) -> float:
    try:
        return max(0.0, min(float(raw), 1.0))
    except (TypeError, ValueError):
        return 0.75


def _normalize_plan_base(
    raw_plan: Dict[str, Any],
    query: str,
    available_tool_names: List[str],
    default_intent: str,
    tool_filter: Optional[Set[str]],
    log_prefix: str,
) -> List[Dict[str, Any]]:
    """Shared core: parse raw_plan tool_calls, validate args, return normalized list."""
    registry = get_tool_registry()
    raw_tool_calls = raw_plan.get("tool_calls") or raw_plan.get("tools") or []
    logger.debug("[%s] normalizing %d raw tool calls", log_prefix, len(raw_tool_calls))
    normalized: List[Dict[str, Any]] = []

    for item in raw_tool_calls:
        if isinstance(item, str):
            name, args = item, {}
        elif isinstance(item, dict):
            name = item.get("name") or item.get("tool")
            args = item.get("args") or item.get("arguments") or {}
        else:
            continue

        if name not in available_tool_names or name not in registry:
            logger.debug("[%s] skipping unknown/unavailable tool '%s'", log_prefix, name)
            continue
        if tool_filter and name not in tool_filter:
            logger.debug("[%s] skipping filtered-out tool '%s'", log_prefix, name)
            continue
        if not isinstance(args, dict):
            args = {}

        signature = inspect.signature(registry[name])
        allowed_args = {k: v for k, v in args.items() if k in signature.parameters}

        # Web-search defaults
        if name == "web_search":
            allowed_args.setdefault("query", query)
        elif name == "scrape_url" and not allowed_args.get("url"):
            url_match = re.search(r"https?://[^\s)>\]]+", query)
            if not url_match:
                logger.debug("[%s] scrape_url skipped — no URL in query", log_prefix)
                continue
            allowed_args["url"] = url_match.group(0).rstrip(".,")

        missing = [
            p for p, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty and p not in allowed_args
        ]
        if missing:
            logger.warning("%s skipped tool '%s'; missing args=%s", log_prefix, name, missing)
            continue

        logger.debug("[%s] accepted tool '%s' args=%s", log_prefix, name, allowed_args)
        normalized.append({"name": name, "args": allowed_args})

    logger.debug("[%s] normalized %d/%d tool calls", log_prefix, len(normalized), len(raw_tool_calls))
    return normalized


# ---------------------------------------------------------------------------
# General assistant plan
# ---------------------------------------------------------------------------

def fallback_tool_plan(query: str, available_tool_names: List[str]) -> Dict[str, Any]:
    tool_calls = (
        [{"name": "web_search", "args": {"query": query}}]
        if "web_search" in available_tool_names else []
    )
    logger.debug("[normalize] using fallback_tool_plan web_search=%s", bool(tool_calls))
    return {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": tool_calls, "routing_source": "fallback"}


def normalize_tool_plan(
    raw_plan: Dict[str, Any],
    query: str,
    available_tool_names: List[str],
) -> Dict[str, Any]:
    intent = str(raw_plan.get("intent") or "GENERAL_QUERY").upper()
    confidence = _parse_confidence(raw_plan.get("confidence", 0.75))
    logger.debug("[normalize_tool_plan] intent=%s confidence=%s", intent, confidence)

    normalized_calls = _normalize_plan_base(
        raw_plan, query, available_tool_names,
        default_intent=intent, tool_filter=None,
        log_prefix="AutoGen router",
    )
    if not normalized_calls:
        return fallback_tool_plan(query, available_tool_names)

    return {"intent": intent, "confidence": confidence, "tool_calls": normalized_calls, "routing_source": "llm"}


# ---------------------------------------------------------------------------
# Travel plan
# ---------------------------------------------------------------------------

TRAVEL_TOOL_NAMES: Set[str] = {
    "search_flights", "search_hotels", "estimate_trip_budget",
    "search_places", "search_restaurants", "generate_itinerary",
    "get_local_transport_info", "get_distance_between_places",
    "generate_trip_summary", "get_currency_exchange", "get_geo_distance",
    "get_weather",
}


def fallback_travel_plan(query: str, available_tool_names: List[str]) -> Dict[str, Any]:
    tool_calls = (
        [{"name": "generate_trip_summary", "args": {"destination": query, "days": "3", "budget": ""}}]
        if "generate_trip_summary" in available_tool_names else []
    )
    logger.debug("[normalize] using fallback_travel_plan trip_summary=%s", bool(tool_calls))
    return {
        "intent": "GENERAL_TRAVEL_QUERY", "confidence": 0.0,
        "entities": {}, "tool_calls": tool_calls, "routing_source": "fallback",
    }


def normalize_travel_tool_plan(
    raw_plan: Dict[str, Any],
    query: str,
    available_tool_names: List[str],
) -> Dict[str, Any]:
    intent = str(raw_plan.get("intent") or "GENERAL_TRAVEL_QUERY").upper()
    confidence = _parse_confidence(raw_plan.get("confidence", 0.75))
    logger.debug("[normalize_travel_tool_plan] intent=%s confidence=%s", intent, confidence)

    normalized_calls = _normalize_plan_base(
        raw_plan, query, available_tool_names,
        default_intent=intent, tool_filter=TRAVEL_TOOL_NAMES,
        log_prefix="AutoGen travel router",
    )
    if not normalized_calls:
        return fallback_travel_plan(query, available_tool_names)

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": raw_plan.get("entities") or {},
        "tool_calls": normalized_calls,
        "routing_source": "llm",
    }
