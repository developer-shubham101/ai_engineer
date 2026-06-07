"""Tool resolution, catalog building, caching, and execution utilities."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .tool_registry import get_tool_registry

logger = logging.getLogger(__name__)


def build_tool_catalog(available_tool_names: List[str]) -> List[Dict[str, Any]]:
    """Build a JSON-serialisable catalog of tools for LLM selector prompts."""
    registry = get_tool_registry()
    return [
        {
            "name": name,
            "description": inspect.getdoc(func) or "",
            "parameters": [
                {
                    "name": param_name,
                    "required": param.default is inspect.Parameter.empty,
                    "default": None if param.default is inspect.Parameter.empty else param.default,
                }
                for param_name, param in inspect.signature(func).parameters.items()
            ],
        }
        for name in available_tool_names
        if (func := registry.get(name))
    ]


def resolve_tools(requested: List[str]) -> List[Callable]:
    """Return tool callables for the requested names (empty list = all tools)."""
    registry = get_tool_registry()
    names = requested if requested else list(registry.keys())
    logger.debug("[resolve_tools] requested=%s resolved_from=%d available", requested, len(registry))
    tools = []
    for name in names:
        if name in registry:
            tools.append(registry[name])
        else:
            logger.warning("AutoGen: unknown tool '%s' requested, skipping", name)
    logger.debug("[resolve_tools] resolved %d tools: %s", len(tools), [t.__name__ for t in tools])
    return tools


def resolve_agent_tools(names: List[str]) -> List[Callable]:
    """Wrap registry callables so AutoGen agents can invoke them by name."""
    registry = get_tool_registry()
    agent_tools = []
    for name in names:
        func = registry.get(name)
        if not func:
            continue

        def _make(tool_name: str, tool_func: Callable) -> Callable:
            def agent_tool(**kwargs):
                return tool_func(**kwargs)
            agent_tool.__name__ = tool_name
            agent_tool.__doc__ = inspect.getdoc(tool_func) or ""
            agent_tool.__signature__ = inspect.signature(tool_func)
            agent_tool.__annotations__ = getattr(tool_func, "__annotations__", {})
            return agent_tool

        agent_tools.append(_make(name, func))
    return agent_tools


def coerce_tool_args(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce arg values to the types declared in the function signature."""
    signature = inspect.signature(func)
    normalized: Dict[str, Any] = {}
    for name, value in args.items():
        param = signature.parameters.get(name)
        if not param:
            continue
        annotation = param.annotation
        try:
            if annotation is int:
                value = int(value)
            elif annotation is float:
                value = float(value)
            elif annotation is str:
                value = str(value)
        except (TypeError, ValueError):
            pass
        normalized[name] = value
    return normalized


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def tool_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name: str,
    func: Callable,
    args: Dict[str, Any],
    cache: Any,  # SemanticCache or plain dict
) -> Dict[str, Any]:
    """Execute a single tool with semantic caching. Returns a result envelope."""
    # Support both SemanticCache (get/set API) and plain dict (legacy)
    if hasattr(cache, "get") and hasattr(cache, "set") and not isinstance(cache, dict):
        cached_result = cache.get(tool_name, args)
        if cached_result is not None:
            logger.debug("[execute_tool] semantic cache HIT tool=%s args=%s", tool_name, args)
            return {"tool": tool_name, "args": args, "result": cached_result, "duration_ms": 0.0, "cached": True}
    else:
        key = tool_cache_key(tool_name, args)
        if key in cache:
            logger.debug("[execute_tool] cache HIT tool=%s args=%s", tool_name, args)
            return {"tool": tool_name, "args": args, "result": cache[key], "duration_ms": 0.0, "cached": True}

    logger.debug("[execute_tool] START tool=%s args=%s", tool_name, args)
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(func, **args)
    except Exception as exc:
        logger.warning("[execute_tool] FAILED tool=%s error=%s", tool_name, exc)
        result = {"status": "error", "error": str(exc)}
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.debug("[execute_tool] DONE tool=%s duration_ms=%s cached=False", tool_name, duration_ms)

    if isinstance(result, dict):
        if hasattr(cache, "set") and not isinstance(cache, dict):
            cache.set(tool_name, args, result)
        else:
            cache[tool_cache_key(tool_name, args)] = result

    return {"tool": tool_name, "args": args, "result": result, "duration_ms": duration_ms, "cached": False}


async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute a list of tool calls in parallel, returning result envelopes."""
    registry = get_tool_registry()
    tasks = [
        execute_tool(tc["name"], func, tc.get("args", {}), cache)
        for tc in tool_calls
        if (func := registry.get(tc["name"]))
    ]
    logger.debug("[execute_tool_calls] running %d tools in parallel", len(tasks))
    if not tasks:
        return []
    results = list(await asyncio.gather(*tasks))
    logger.debug("[execute_tool_calls] all done: %s", [r["tool"] for r in results])
    return results
