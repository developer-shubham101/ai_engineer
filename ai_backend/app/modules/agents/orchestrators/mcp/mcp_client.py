from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:8000/sse"


class MCPClient:
    """
    HTTP/SSE MCP Client

    Connects to external MCP server running on port 8000.
    """

    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url

    async def list_tools(self) -> List[Dict[str, Any]]:

        async with sse_client(self.server_url) as (read, write):

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

                logger.info("[MCP] Loaded %d tools", len(catalog))

                return catalog

    async def call_tool(
        self,
        name: str,
        args: Dict[str, Any],
    ) -> Any:

        logger.info("[MCP] Calling tool=%s args=%s", name, args)

        async with sse_client(self.server_url) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                response = await session.call_tool(name, args)

                if response.content:

                    raw = (
                        response.content[0].text
                        if hasattr(response.content[0], "text")
                        else str(response.content[0])
                    )

                    try:
                        return json.loads(raw)

                    except Exception:
                        return raw

                return {}

    async def call_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        import time

        async def _one(tc: Dict[str, Any]):

            start = time.perf_counter()

            try:

                result = await self.call_tool(
                    tc["name"],
                    tc.get("args", {}),
                )

            except Exception as exc:

                logger.exception("Tool failed")

                result = {
                    "status": "error",
                    "error": str(exc),
                }

            duration_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            return {
                "tool": tc["name"],
                "args": tc.get("args", {}),
                "result": result,
                "duration_ms": duration_ms,
                "cached": False,
            }

        return await asyncio.gather(
            *[_one(tc) for tc in tool_calls]
        )