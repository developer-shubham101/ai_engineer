"""Smart Travel Planner workflow.

A lightweight, tool-driven travel assistant that:
  1. Classifies travel intent via keyword matching
  2. Extracts travel entities (destination, days, budget, etc.)
  3. Selects only the relevant tools for that intent
  4. Executes tools (in parallel where possible)
  5. Aggregates results into a structured travel plan

No new agent ecosystems — reuses existing tool functions and async patterns.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ...function_tools.tool_weather import get_weather
from ...function_tools.tool_web_search import web_search
from ...function_tools.tool_travel import (
    estimate_trip_budget,
    generate_itinerary,
    generate_trip_summary,
    get_distance_between_places,
    get_local_transport_info,
    search_flights,
    search_hotels,
    search_places,
    search_restaurants,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------

TRAVEL_INTENTS = [
    "FLIGHT_SEARCH",
    "HOTEL_SEARCH",
    "ITINERARY_PLANNING",
    "BUDGET_TRAVEL",
    "WEATHER_TRAVEL",
    "LOCAL_ATTRACTIONS",
    "RESTAURANT_SEARCH",
    "TRANSPORT_QUERY",
    "GENERAL_TRAVEL_QUERY",
]

# keyword → intent (first match wins; order matters)
_INTENT_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("FLIGHT_SEARCH",       ["flight", "fly", "airline", "airfare", "plane ticket"]),
    ("HOTEL_SEARCH",        ["hotel", "stay", "accommodation", "hostel", "resort", "lodge"]),
    ("WEATHER_TRAVEL",      ["weather", "climate", "temperature", "rain", "monsoon", "season"]),
    ("RESTAURANT_SEARCH",   ["restaurant", "food", "eat", "dining", "cuisine", "cafe"]),
    ("TRANSPORT_QUERY",     ["transport", "bus", "train", "cab", "taxi", "auto", "commute", "how to reach"]),
    ("LOCAL_ATTRACTIONS",   ["places", "attractions", "sightseeing", "tourist", "visit", "things to do", "explore"]),
    ("BUDGET_TRAVEL",       ["budget", "cheap", "affordable", "under", "cost", "expense", "₹", "inr", "rupee"]),
    ("ITINERARY_PLANNING",  ["itinerary", "plan", "day", "schedule", "trip plan", "days"]),
    ("GENERAL_TRAVEL_QUERY", []),   # catch-all — always matches last
]

# intent → tools to run
_INTENT_TOOLS: Dict[str, List[str]] = {
    "FLIGHT_SEARCH":       ["search_flights"],
    "HOTEL_SEARCH":        ["search_hotels"],
    "WEATHER_TRAVEL":      ["get_weather"],
    "RESTAURANT_SEARCH":   ["search_restaurants"],
    "TRANSPORT_QUERY":     ["get_local_transport_info"],
    "LOCAL_ATTRACTIONS":   ["search_places", "web_search"],
    "BUDGET_TRAVEL":       ["estimate_trip_budget", "search_hotels", "get_weather"],
    "ITINERARY_PLANNING":  ["generate_itinerary", "search_places", "get_weather", "estimate_trip_budget"],
    "GENERAL_TRAVEL_QUERY": ["web_search", "generate_trip_summary"],
}


# ---------------------------------------------------------------------------
# Step 1 — Intent classification
# ---------------------------------------------------------------------------

def classify_travel_intent(query: str) -> str:
    """Classify travel intent using keyword matching."""
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if not keywords:          # catch-all
            return intent
        if any(kw in q for kw in keywords):
            return intent
    return "GENERAL_TRAVEL_QUERY"


# ---------------------------------------------------------------------------
# Step 2 — Entity extraction
# ---------------------------------------------------------------------------

def extract_travel_entities(query: str) -> Dict[str, Any]:
    """Extract travel entities from a natural language query."""
    q = query.lower()

    # --- destination ---
    known_destinations = [
        "goa", "jaipur", "kerala", "mumbai", "delhi", "bangalore", "hyderabad",
        "manali", "shimla", "udaipur", "agra", "varanasi", "kolkata", "chennai",
        "paris", "london", "dubai", "singapore", "bangkok", "bali", "new york",
    ]
    destination = next((d.title() for d in known_destinations if d in q), None)

    # --- number of days ---
    days_match = re.search(r"(\d+)\s*(?:day|days|night|nights)", q)
    days = int(days_match.group(1)) if days_match else 3

    # --- budget ---
    budget_match = re.search(r"(?:under|below|within|budget|₹|rs\.?|inr)\s*(\d[\d,]*)", q)
    budget = int(budget_match.group(1).replace(",", "")) if budget_match else None

    # --- source location ---
    from_match = re.search(r"from\s+([a-z]+(?:\s+[a-z]+)?)", q)
    source = from_match.group(1).title() if from_match else None

    # --- number of travelers ---
    travelers_match = re.search(r"(\d+)\s*(?:person|people|traveler|travellers|pax)", q)
    travelers = int(travelers_match.group(1)) if travelers_match else 1

    return {
        "destination": destination,
        "days": days,
        "budget": budget,
        "source": source,
        "travelers": travelers,
    }


# ---------------------------------------------------------------------------
# Step 3 — Tool selection
# ---------------------------------------------------------------------------

def select_travel_tools(query: str) -> Tuple[str, List[str]]:
    """Classify intent and return the filtered tool list for that intent."""
    intent = classify_travel_intent(query)
    tools = _INTENT_TOOLS.get(intent, ["web_search"])
    logger.debug("TRAVEL_PLANNER: intent=%s | tools=%s", intent, tools)
    return intent, tools


# ---------------------------------------------------------------------------
# Step 4 — Tool execution (async, parallel where safe)
# ---------------------------------------------------------------------------

async def _run_tool(name: str, entities: Dict[str, Any]) -> Tuple[str, Any]:
    """Execute a single travel tool and return (name, result)."""
    dest = entities.get("destination") or "the destination"
    days = str(entities.get("days", 3))
    budget = str(entities.get("budget", ""))
    source = entities.get("source") or "Delhi"
    travelers = str(entities.get("travelers", 1))

    loop = asyncio.get_event_loop()

    tool_map = {
        "search_flights":          lambda: search_flights(source, dest, budget=budget),
        "search_hotels":           lambda: search_hotels(dest, budget=budget, days=days),
        "get_weather":             lambda: get_weather(dest),
        "search_restaurants":      lambda: search_restaurants(dest),
        "get_local_transport_info":lambda: get_local_transport_info(dest),
        "search_places":           lambda: search_places(dest),
        "estimate_trip_budget":    lambda: estimate_trip_budget(dest, days=days, travelers=travelers),
        "generate_itinerary":      lambda: generate_itinerary(dest, days=days, budget=budget),
        "generate_trip_summary":   lambda: generate_trip_summary(dest, days=days, budget=budget),
        "get_distance_between_places": lambda: get_distance_between_places(source, dest),
        "web_search":              lambda: web_search(f"travel guide {dest}"),
    }

    fn = tool_map.get(name)
    if fn is None:
        return name, {"error": f"Unknown tool: {name}", "status": "error"}

    try:
        # All tool functions are sync — run in executor to avoid blocking
        result = await loop.run_in_executor(None, fn)
        return name, result
    except Exception as e:
        logger.warning("TRAVEL_PLANNER: tool %s failed | error=%s", name, e)
        return name, {"error": str(e), "status": "error"}


async def execute_travel_tools(tools: List[str], entities: Dict[str, Any]) -> Dict[str, Any]:
    """Execute all selected tools in parallel and return aggregated results."""
    tasks = [_run_tool(name, entities) for name in tools]
    pairs = await asyncio.gather(*tasks)
    return {name: result for name, result in pairs}


# ---------------------------------------------------------------------------
# Step 5 — Build structured travel plan
# ---------------------------------------------------------------------------

def build_travel_plan(
    query: str,
    intent: str,
    entities: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate tool results into a structured travel plan response."""
    dest = entities.get("destination") or "Unknown"
    days = entities.get("days", 3)
    budget = entities.get("budget")

    plan: Dict[str, Any] = {
        "query": query,
        "intent": intent,
        "destination": dest,
        "duration_days": days,
    }

    if budget:
        plan["budget_inr"] = budget

    # Attach each tool result under a clean key
    key_map = {
        "get_weather":              "weather",
        "search_flights":           "flights",
        "search_hotels":            "recommended_hotels",
        "search_restaurants":       "restaurants",
        "search_places":            "attractions",
        "estimate_trip_budget":     "estimated_cost",
        "generate_itinerary":       "itinerary",
        "generate_trip_summary":    "trip_summary",
        "get_local_transport_info": "local_transport",
        "get_distance_between_places": "distance_info",
        "web_search":               "web_results",
    }

    for tool_name, result in tool_results.items():
        key = key_map.get(tool_name, tool_name)
        plan[key] = result

    # Flatten itinerary to day-wise list for readability
    if "itinerary" in plan and isinstance(plan["itinerary"], dict):
        plan["itinerary"] = plan["itinerary"].get("itinerary", plan["itinerary"])

    # Pull travel tips from summary if present
    summary = plan.get("trip_summary", {})
    if isinstance(summary, dict) and "travel_tips" in summary:
        plan["travel_tips"] = summary.pop("travel_tips")

    return plan


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_smart_travel_planner(topic: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Execute the smart_travel_planner workflow.

    Returns:
        (structured_plan, agents_used)
    """
    logger.info("TRAVEL_PLANNER: start | query=%r", topic)
    start = time.time()

    # Step 1 + 2
    intent, tools = select_travel_tools(topic)
    entities = extract_travel_entities(topic)

    logger.info(
        "TRAVEL_PLANNER: intent=%s | destination=%s | days=%s | budget=%s | tools=%s",
        intent, entities["destination"], entities["days"], entities["budget"], tools,
    )

    # Step 3 + 4 — parallel tool execution
    tool_results = await execute_travel_tools(tools, entities)

    # Step 5 — structured plan
    plan = build_travel_plan(topic, intent, entities, tool_results)
    plan["_meta"] = {
        "tools_used": tools,
        "processing_ms": int((time.time() - start) * 1000),
    }

    agents_used = [f"TravelPlanner({intent})"] + [f"Tool:{t}" for t in tools]
    logger.info("TRAVEL_PLANNER: done | tools=%s | ms=%d", tools, plan["_meta"]["processing_ms"])

    return plan, agents_used
