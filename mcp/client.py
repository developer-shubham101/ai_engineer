"""MCP client — connects to server.py via stdio and calls tools interactively."""
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_CMD = StdioServerParameters(command=sys.executable, args=["server.py"])


async def list_tools(session: ClientSession) -> None:
    tools = await session.list_tools()
    print("\nAvailable tools:")
    for t in tools.tools:
        print(f"  {t.name}: {t.description or ''}")


async def call_tool(session: ClientSession, name: str, args: dict) -> None:
    result = await session.call_tool(name, args)
    for content in result.content:
        print(content.text if hasattr(content, "text") else content)


async def repl(session: ClientSession) -> None:
    await list_tools(session)
    print("\nEnter: <tool_name> <json_args>  |  'list' to refresh  |  'quit' to exit\n")
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, input, "> ")
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            break
        if line == "list":
            await list_tools(session)
            continue
        parts = line.split(None, 1)
        tool_name = parts[0]
        args = json.loads(parts[1]) if len(parts) > 1 else {}
        try:
            await call_tool(session, tool_name, args)
        except Exception as e:
            print(f"Error: {e}")


async def main() -> None:
    async with stdio_client(SERVER_CMD) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await repl(session)


if __name__ == "__main__":
    asyncio.run(main())
