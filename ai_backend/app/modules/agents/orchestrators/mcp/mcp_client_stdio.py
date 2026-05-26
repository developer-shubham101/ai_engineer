"""Async MCP client — thin wrapper around the mcp/server.py stdio transport.

Learning note
-------------
AutoGen orchestrator calls tools like this:
    result = await asyncio.to_thread(func, **args)   # local Python function

This client replaces that with:
    result = await mcp_client.call_tool(name, args)  # MCP over stdio

Everything else in the pipeline (ToolSelector, Summarizer, step merging) is
identical to the AutoGen orchestrator.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Path to mcp/server.py — adjust if your layout differs
_MCP_SERVER = Path(__file__).parents[8] / "mcp" / "server.py"
_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(_MCP_SERVER)],
    cwd=str(_MCP_SERVER.parent),
)


class MCPClient:
    """Stateless async MCP client.

    Each public method opens a fresh stdio connection, calls the server,
    and closes the connection.  This is simple and safe for learning; in
    production you would keep a persistent session.

    Learning note — why a new connection per call?
    -----------------------------------------------
    The MCP stdio transport is a child process.  Keeping it alive across
    requests requires careful lifecycle management.  For this educational
    implementation we keep it simple: one call = one process.
    """

    def __init__(self, server_params: Optional[StdioServerParameters] = None) -> None:
        self._params = server_params or _SERVER_PARAMS

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return a list of tool descriptors from the MCP server.

        Each descriptor mirrors the AutoGen tool catalog shape:
            {"name": str, "description": str, "parameters": [...]}

        Learning note — AutoGen equivalent:
            build_tool_catalog(available_tool_names)  # from tool_utils.py
        """
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                catalog = []
                for t in response.tools:
                    params = []
                    if t.inputSchema and "properties" in t.inputSchema:
                        required = t.inputSchema.get("required", [])
                        for param_name, schema in t.inputSchema["properties"].items():
                            params.append({
                                "name": param_name,
                                "required": param_name in required,
                                "default": schema.get("default"),
                                "description": schema.get("description", ""),
                            })
                    catalog.append({
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": params,
                    })
                logger.debug("[MCPClient] list_tools returned %d tools", len(catalog))
                return catalog

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Call a single tool on the MCP server and return its result.

        Learning note — AutoGen equivalent:
            result = await asyncio.to_thread(func, **args)  # from tool_utils.py
        """
        logger.debug("[MCPClient] call_tool name=%s args=%s", name, args)
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(name, args)
                # MCP returns a list of content blocks; extract text from first
                if response.content:
                    raw = response.content[0].text if hasattr(response.content[0], "text") else str(response.content[0])
                    try:
                        return json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        return raw
                return {}

    async def call_tools_parallel(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls and return result envelopes.

        Shape of each envelope (same as AutoGen execute_tool_calls output):
            {"tool": str, "args": dict, "result": any, "duration_ms": float, "cached": False}

        Learning note — AutoGen equivalent:
            execute_tool_calls(tool_calls, cache)  # from tool_utils.py
        """
        import asyncio
        import time

        async def _one(tc: Dict[str, Any]) -> Dict[str, Any]:
            start = time.perf_counter()
            try:
                result = await self.call_tool(tc["name"], tc.get("args", {}))
            except Exception as exc:
                logger.warning("[MCPClient] tool %s failed: %s", tc["name"], exc)
                result = {"status": "error", "error": str(exc)}
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "tool": tc["name"],
                "args": tc.get("args", {}),
                "result": result,
                "duration_ms": duration_ms,
                "cached": False,
            }

        results = list(await asyncio.gather(*[_one(tc) for tc in tool_calls]))
        logger.debug("[MCPClient] parallel results: %s", [r["tool"] for r in results])
        return results
