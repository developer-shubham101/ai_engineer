"""MCP-backed orchestrator — same 3-agent pipeline as AutoGenOrchestrator.

Learning map (AutoGen → MCP)
-----------------------------
AutoGen                             MCP equivalent
-----------------------------------+------------------------------------------
tool_registry.get_tool_registry()  | mcp_client.list_tools()
tool_utils.execute_tool_calls()    | mcp_client.call_tools_parallel()
asyncio.to_thread(func, **args)    | mcp_client.call_tool(name, args)
build_tool_catalog(names)          | mcp_client.list_tools() (already catalog shape)

Everything else is identical:
  - ToolSelector  AssistantAgent  (max 2 steps)
  - ToolExecutor  deterministic via MCPClient
  - Summarizer    AssistantAgent  (max = user max_steps)
  - step merging, debug_info, AgentResponse shape
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


class MCPOrchestrator(IAgentOrchestrator):
    """AutoGen 3-agent pipeline that calls tools via MCP instead of local imports.

    Workflow: smart_assistant only (for learning focus).
    Extend with debate/research/travel by following the same pattern.

    Learning note
    -------------
    The only difference from AutoGenOrchestrator.execute_smart_assistant_workflow:

        AutoGen:  tool_results = await execute_tool_calls(tool_calls, cache)
        MCP:      tool_results = await self._mcp.call_tools_parallel(tool_calls)

    The ToolSelector and Summarizer agents are byte-for-byte identical.
    """

    AVAILABLE_WORKFLOWS = ["smart_assistant"]

    def __init__(self, model_client: Any) -> None:
        self.model_client = model_client
        self._mcp = MCPClient()
        # Cache of tool catalog so we don't hit the MCP server on every request
        self._tool_catalog: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    def register_tool(self, tool: Any) -> None:
        pass  # Tools come from MCP server, not registered here

    def get_available_tools(self) -> List[str]:
        # Sync stub — real list fetched async in _get_catalog()
        return []

    def get_available_workflows(self) -> List[str]:
        return self.AVAILABLE_WORKFLOWS

    async def process_request(
        self, request: AgentRequest, user: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        workflow = request.workflow.lower()
        if workflow != "smart_assistant":
            return AgentResponse(
                answer=f"MCPOrchestrator only supports 'smart_assistant'. Got: '{workflow}'",
                steps=[], tools_used=[], final_step=True,
            )
        try:
            return await self._run_smart_assistant(request.question, request.max_steps)
        except Exception as exc:
            logger.error("[MCPOrchestrator] workflow failed: %s", exc, exc_info=True)
            return AgentResponse(answer=f"Workflow failed: {exc}", steps=[], tools_used=[], final_step=True)

    # ------------------------------------------------------------------
    # Internal helpers  
    # ------------------------------------------------------------------

    async def _get_catalog(self) -> List[Dict[str, Any]]:
        """Fetch tool catalog from MCP server (cached after first call).

        Learning note — AutoGen equivalent:
            build_tool_catalog(available_tool_names)  # tool_utils.py
        """
        if self._tool_catalog is None:
            self._tool_catalog = await self._mcp.list_tools()
            logger.debug("[MCPOrchestrator] fetched %d tools from MCP", len(self._tool_catalog))
        return self._tool_catalog

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from LLM output."""
        import re
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE).replace("```", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_plan(
        self,
        raw: Dict[str, Any],
        query: str,
        available_names: List[str],
    ) -> Dict[str, Any]:
        """Validate and clean the ToolSelector JSON output.

        Learning note — AutoGen equivalent:
            normalize_tool_plan(parsed, query, available_tool_names)  # plan_normalizer.py
        """
        intent = str(raw.get("intent") or "GENERAL_QUERY").upper()
        try:
            confidence = max(0.0, min(float(raw.get("confidence", 0.75)), 1.0))
        except (TypeError, ValueError):
            confidence = 0.75

        normalized = []
        for item in (raw.get("tool_calls") or raw.get("tools") or []):
            name = item.get("name") if isinstance(item, dict) else item
            args = item.get("args", {}) if isinstance(item, dict) else {}
            if name not in available_names:
                continue
            normalized.append({"name": name, "args": args if isinstance(args, dict) else {}})

        if not normalized:
            fallback = [{"name": "web_search", "args": {"query": query}}] if "web_search" in available_names else []
            return {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": fallback, "routing_source": "fallback"}

        return {"intent": intent, "confidence": confidence, "tool_calls": normalized, "routing_source": "llm"}

    @staticmethod
    def _build_executor_steps(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool result envelopes into step records.

        Learning note — AutoGen equivalent:
            build_executor_steps(tool_results)  # step_utils.py
        """
        return [
            {
                "step": idx,
                "agent": "ToolExecutor",
                "type": "tool_execution",
                "tool": r["tool"],
                "args": r["args"],
                "content": json.dumps(r["result"], indent=2, default=str),
                "duration_ms": r.get("duration_ms"),
                "cached": r.get("cached"),
            }
            for idx, r in enumerate(tool_results, start=1)
        ]

    # ------------------------------------------------------------------
    # 3-agent smart_assistant workflow
    # ------------------------------------------------------------------

    async def _run_smart_assistant(self, query: str, max_steps: int) -> AgentResponse:
        """
        3-agent pipeline (identical structure to AutoGen smart_assistant):

          Agent 1 — ToolSelector   : picks tools from MCP catalog  (max 2 steps)
          Agent 2 — ToolExecutor   : calls tools via MCPClient      (deterministic)
          Agent 3 — Summarizer     : formats final answer           (max = max_steps)
        """
        catalog = await self._get_catalog()
        available_names = [t["name"] for t in catalog if t["name"] != "save_research_report"]

        # ── Agent 1: ToolSelector ─────────────────────────────────────────────
        # Learning note: identical prompt + agent as AutoGen ToolSelector.
        # Only difference: catalog comes from MCP, not build_tool_catalog().
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
            model_client=self.model_client,
        )
        selector_team = RoundRobinGroupChat(
            participants=[selector],
            termination_condition=MaxMessageTermination(max_messages=2),
        )
        selector_task = (
            f"Tool catalog:\n{json.dumps(catalog, indent=2, default=str)}\n\n"
            f"User query: {query}\n\n"
            "Return only the JSON tool plan."
        )

        selector_result, selector_steps = "", []
        try:
            async for msg in selector_team.run_stream(task=selector_task):
                if hasattr(msg, "content") and msg.content:
                    selector_result = str(msg.content).strip()
                    selector_steps.append({
                        "step": len(selector_steps) + 1,
                        "agent": getattr(msg, "source", "ToolSelector"),
                        "type": "reasoning",
                        "content": selector_result,
                    })
            parsed = self._extract_json(selector_result)
            if not parsed:
                raise ValueError(f"ToolSelector did not return JSON: {selector_result!r}")
            route_plan = self._normalize_plan(parsed, query, available_names)
        except Exception as exc:
            logger.warning("[MCPOrchestrator] ToolSelector failed, falling back: %s", exc)
            fallback_calls = [{"name": "web_search", "args": {"query": query}}] if "web_search" in available_names else []
            route_plan = {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": fallback_calls, "routing_source": "fallback"}
            selector_steps = []

        intent = route_plan["intent"]
        confidence = route_plan["confidence"]
        tool_calls = route_plan["tool_calls"]
        selected_names = [tc["name"] for tc in tool_calls]

        if not tool_calls and "web_search" in available_names:
            tool_calls = [{"name": "web_search", "args": {"query": query}}]
            selected_names = ["web_search"]

        if not selector_steps:
            selector_steps = [{
                "step": 1, "agent": "ToolSelector", "type": "tool_routing",
                "content": json.dumps(
                    {"intent": intent, "confidence": confidence,
                     "routing_source": route_plan.get("routing_source"),
                     "tool_calls": tool_calls},
                    default=str,
                ),
            }]

        logger.debug("[MCPOrchestrator] intent=%s tools=%s", intent, selected_names)

        # ── Agent 2: ToolExecutor (via MCP) ───────────────────────────────────
        # Learning note: AutoGen calls asyncio.to_thread(func, **args).
        #                Here we call self._mcp.call_tools_parallel(tool_calls).
        #                The result envelope shape is identical.
        tool_results = await self._mcp.call_tools_parallel(tool_calls)
        executor_steps = self._build_executor_steps(tool_results)
        executor_tools_used = {r["tool"] for r in tool_results}
        executor_result = json.dumps(tool_results, indent=2, default=str)

        # ── Agent 3: Summarizer ───────────────────────────────────────────────
        # Learning note: byte-for-byte identical to AutoGen Summarizer agent.
        summarizer = AssistantAgent(
            name="Summarizer",
            system_message=(
                "You are the final assistant. Tool results are already provided — do not call any tools. "
                "Summarize the results clearly and concisely. "
                "When the answer contains multiple independent facts (e.g. weather + stock price), "
                "return plain text always in formatted way so user can read."
            ),
            model_client=self.model_client,
        )
        summarizer_team = RoundRobinGroupChat(
            participants=[summarizer],
            termination_condition=MaxMessageTermination(max_messages=max_steps),
        )
        summarizer_task = (
            f"User query: {query}\n"
            f"Detected intent: {intent}\n"
            f"Tools used: {json.dumps(selected_names)}\n"
            f"Tool results:\n{executor_result}"
        )

        final_result, summary_steps, summary_tools_used = "", [], set()
        async for msg in summarizer_team.run_stream(task=summarizer_task):
            if hasattr(msg, "content") and msg.content:
                content_str = str(msg.content).strip()
                if content_str:
                    final_result = content_str
                    summary_steps.append({
                        "step": len(summary_steps) + 1,
                        "agent": getattr(msg, "source", "Summarizer"),
                        "type": "reasoning",
                        "content": content_str,
                    })

        # ── Merge steps with sequential numbering ─────────────────────────────
        # Learning note: identical to AutoGen merge_steps() in step_utils.py
        pre_summary = selector_steps + executor_steps
        for i, step in enumerate(pre_summary, start=1):
            step["step"] = i
        for step in summary_steps:
            step["step"] = step.get("step", 0) + len(pre_summary)

        tools_used = (executor_tools_used or set(selected_names)) | summary_tools_used

        return AgentResponse(
            answer=final_result,
            steps=pre_summary + summary_steps,
            tools_used=list(tools_used),
            final_step=True,
            debug_info={
                "intent": intent,
                "confidence": confidence,
                "selected_tools": selected_names,
                "routing_source": route_plan.get("routing_source"),
                "tool_calls": tool_calls,
                "transport": "mcp_stdio",          # <-- only new field vs AutoGen
            },
        )
