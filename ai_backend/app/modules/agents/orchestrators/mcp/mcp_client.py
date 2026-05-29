"""HTTP/SSE MCP client — connects to an external MCP server and exposes list_tools / call_tool / call_tools_parallel."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:4154/sse"


class MCPClient:
    """Thin async client over the MCP HTTP/SSE transport.

    All three public methods open a fresh SSE connection per call so the
    client stays stateless and safe to share across concurrent requests.
    """

    def __init__(self, server_url: str = MCP_SERVER_URL) -> None:
        self.server_url = server_url
        logger.debug("[MCPClient] initialized server_url=%s", self.server_url)

    # ------------------------------------------------------------------
    # list_tools
    # ------------------------------------------------------------------

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch the tool catalog from the MCP server.

        Returns a list of dicts compatible with build_tool_catalog() shape:
            [{"name": str, "description": str, "parameters": [...]}]
        """
        logger.debug("[MCPClient.list_tools] connecting to %s", self.server_url)
        try:
            async with sse_client(self.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.debug("[MCPClient.list_tools] session initialized")

                    response = await session.list_tools()
                    logger.debug("[MCPClient.list_tools] raw tool count=%d", len(response.tools))

                    catalog: List[Dict[str, Any]] = []
                    for t in response.tools:
                        params: List[Dict[str, Any]] = []
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
                        logger.debug(
                            "[MCPClient.list_tools] tool=%s params=%d",
                            t.name, len(params),
                        )

                    logger.info("[MCPClient.list_tools] loaded %d tools from %s", len(catalog), self.server_url)
                    return catalog

        except Exception as exc:
            logger.exception("[MCPClient.list_tools] failed to fetch tools from %s: %s", self.server_url, exc)
            raise

    # ------------------------------------------------------------------
    # call_tool
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Call a single tool on the MCP server and return its parsed result.

        Returns a dict on success, a plain string if the response is not
        valid JSON, or an empty dict if the server returned no content.
        """
        logger.debug("[MCPClient.call_tool] START tool=%s args=%s", name, args)
        start = time.perf_counter()

        try:
            async with sse_client(self.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.call_tool(name, args)

                    if not response.content:
                        duration_ms = round((time.perf_counter() - start) * 1000, 2)
                        logger.warning(
                            "[MCPClient.call_tool] tool=%s returned empty content duration_ms=%s",
                            name, duration_ms,
                        )
                        return {}

                    raw = (
                        response.content[0].text
                        if hasattr(response.content[0], "text")
                        else str(response.content[0])
                    )

                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug(
                            "[MCPClient.call_tool] tool=%s response is not JSON, returning raw string",
                            name,
                        )
                        result = raw

                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    logger.info(
                        "[MCPClient.call_tool] DONE tool=%s duration_ms=%s result_type=%s",
                        name, duration_ms, type(result).__name__,
                    )
                    return result

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "[MCPClient.call_tool] FAILED tool=%s duration_ms=%s error=%s",
                name, duration_ms, exc,
            )
            raise

    # ------------------------------------------------------------------
    # call_tools_parallel
    # ------------------------------------------------------------------

    async def call_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls concurrently.

        Returns a list of result envelopes with the same shape as
        execute_tool_calls() in utils/tool_utils.py:
            {"tool": str, "args": dict, "result": Any, "duration_ms": float, "cached": False}
        """
        logger.debug("[MCPClient.call_tools_parallel] START count=%d tools=%s",
                     len(tool_calls), [tc["name"] for tc in tool_calls])

        async def _one(tc: Dict[str, Any]) -> Dict[str, Any]:
            name = tc["name"]
            args = tc.get("args", {})
            start = time.perf_counter()
            try:
                result = await self.call_tool(name, args)
                status = "success"
            except Exception as exc:
                logger.error(
                    "[MCPClient.call_tools_parallel] tool=%s FAILED error=%s", name, exc,
                )
                result = {"status": "error", "error": str(exc)}
                status = "error"

            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.debug(
                "[MCPClient.call_tools_parallel] tool=%s status=%s duration_ms=%s",
                name, status, duration_ms,
            )
            return {
                "tool": name,
                "args": args,
                "result": result,
                "duration_ms": duration_ms,
                "cached": False,
            }

        results = list(await asyncio.gather(*[_one(tc) for tc in tool_calls]))
        logger.info(
            "[MCPClient.call_tools_parallel] DONE count=%d results=%s",
            len(results), [r["tool"] for r in results],
        )
        return results
