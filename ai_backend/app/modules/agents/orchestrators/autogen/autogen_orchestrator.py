"""AutoGen-based agent orchestrator for multi-agent conversations using AutoGen v0.4."""

import asyncio
import inspect
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List, Callable, Set

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

# Registry of all available tool builder functions — names match agent_runner.REGISTRY
_TOOL_BUILDERS: Dict[str, Callable] = {}


def _register_tool_builders() -> Dict[str, Callable]:
    """Lazily build the tool-name → callable map.

    Names are kept in sync with agent_runner.REGISTRY so /tools and
    /tools/{tool_name}/test work for both custom and AutoGen orchestrators.
    """
    if _TOOL_BUILDERS:
        return _TOOL_BUILDERS

    from ...function_tools.tool_web_search import web_search
    from ...function_tools.tool_web_scraper import scrape_url
    from ...function_tools.tool_stock import get_stock_price, get_stock_history, get_crypto_price
    from ...function_tools.tool_weather import get_weather
    from ...function_tools.tool_chart import generate_stock_chart, generate_chart
    from ...function_tools.tool_file import save_research_report
    from ...function_tools.tool_travel import (
        search_flights, search_hotels, estimate_trip_budget, search_places,
        search_restaurants, generate_itinerary, get_local_transport_info,
        get_distance_between_places, generate_trip_summary,
        get_currency_exchange, get_geo_distance,
    )

    def web_search_tool(query: str) -> Dict[str, Any]:
        """Search the internet for real-time information on any topic."""
        return web_search(query, max_results=5)

    def scrape_url_tool(url: str) -> Dict[str, Any]:
        """Fetch and extract full text content from a URL."""
        return scrape_url(url)

    def get_stock_price_tool(symbol: str) -> Dict[str, Any]:
        """Get the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
        return get_stock_price(symbol)

    def get_stock_history_tool(symbol: str, period: str = "5y") -> Dict[str, Any]:
        """Get historical stock prices for a ticker symbol."""
        return get_stock_history(symbol, period)

    def generate_stock_chart_tool(symbol: str, period: str = "5y") -> Dict[str, Any]:
        """Generate a stock performance chart for a symbol over a period."""
        return generate_stock_chart(symbol, period)

    def get_crypto_price_tool(symbol: str) -> Dict[str, Any]:
        """Get the current crypto price for a symbol (e.g. BTC-USD)."""
        return get_crypto_price(symbol)

    def generate_chart_tool(title: str, data: Any, chart_type: str = "line") -> Dict[str, Any]:
        """Generate a generic chart from structured data."""
        return generate_chart(title, data, chart_type)

    def get_weather_tool(city: str) -> Dict[str, Any]:
        """Get current weather conditions for a city."""
        return get_weather(city)

    def save_research_report_tool(
        title: str,
        query: str,
        summary: str,
        markdown: str,
        metadata: str,
        sources: str,
    ) -> str:
        """Save a structured research report as markdown + JSON sidecar.

        Args:
            title:    Report title (used as filename base).
            query:    Original research query.
            summary:  Executive summary (1-3 sentences).
            markdown: Full report body in markdown format.
            metadata: JSON string of extra metadata (tags, topic, etc.).
            sources:  Newline-separated list of source URLs or citations.
        """
        result = save_research_report(title, query, summary, markdown, metadata, sources)
        if result.get("status") == "success":
            return (
                f"Report saved: '{result['title']}' "
                f"({result['size']} chars, {result['sources_count']} sources) "
                f"at {result['report_path']}"
            )
        return f"Save failed: {result.get('error')}"

    def search_flights_tool(origin: str, destination: str, date: str = "", budget: str = "") -> Dict[str, Any]:
        """Search for flights between two cities."""
        return search_flights(origin, destination, date, budget)

    def search_hotels_tool(destination: str, budget: str = "", days: str = "") -> Dict[str, Any]:
        """Search for hotels at a destination."""
        return search_hotels(destination, budget, days)

    def estimate_trip_budget_tool(destination: str, days: str = "3", travelers: str = "1") -> Dict[str, Any]:
        """Estimate total trip budget including flights, hotels, food, and activities."""
        return estimate_trip_budget(destination, days, travelers)

    def search_places_tool(destination: str, category: str = "tourist") -> Dict[str, Any]:
        """Search for tourist attractions and places of interest at a destination."""
        return search_places(destination, category)

    def search_restaurants_tool(destination: str, cuisine: str = "local") -> Dict[str, Any]:
        """Search for restaurants and dining options at a destination."""
        return search_restaurants(destination, cuisine)

    def generate_itinerary_tool(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
        """Generate a day-wise travel itinerary for a destination."""
        return generate_itinerary(destination, days, budget)

    def get_local_transport_info_tool(destination: str) -> Dict[str, Any]:
        """Get local transport options (auto, taxi, bus, rental) at a destination."""
        return get_local_transport_info(destination)

    def get_distance_between_places_tool(origin: str, destination: str) -> Dict[str, Any]:
        """Get approximate distance and travel time between two places."""
        return get_distance_between_places(origin, destination)

    def generate_trip_summary_tool(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
        """Generate a concise trip summary with highlights and travel tips."""
        return generate_trip_summary(destination, days, budget)

    def get_currency_exchange_tool(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
        """Convert amount between currencies using real exchange rates."""
        return get_currency_exchange(from_currency, to_currency, amount)

    def get_geo_distance_tool(origin: str, destination: str) -> Dict[str, Any]:
        """Get real straight-line distance between two places via OpenStreetMap."""
        return get_geo_distance(origin, destination)

    _TOOL_BUILDERS.update({
        "web_search": web_search_tool,
        "scrape_url": scrape_url_tool,
        "get_stock_price": get_stock_price_tool,
        "get_stock_history": get_stock_history_tool,
        "generate_stock_chart": generate_stock_chart_tool,
        "get_crypto_price": get_crypto_price_tool,
        "generate_chart": generate_chart_tool,
        "get_weather": get_weather_tool,
        "save_research_report": save_research_report_tool,
        # Travel tools
        "search_flights": search_flights_tool,
        "search_hotels": search_hotels_tool,
        "estimate_trip_budget": estimate_trip_budget_tool,
        "search_places": search_places_tool,
        "search_restaurants": search_restaurants_tool,
        "generate_itinerary": generate_itinerary_tool,
        "get_local_transport_info": get_local_transport_info_tool,
        "get_distance_between_places": get_distance_between_places_tool,
        "generate_trip_summary": generate_trip_summary_tool,
        "get_currency_exchange": get_currency_exchange_tool,
        "get_geo_distance": get_geo_distance_tool,
    })
    return _TOOL_BUILDERS


class AutoGenOrchestrator(IAgentOrchestrator):
    """AutoGen-based multi-agent orchestrator using v0.4 API.

    Workflow and tools are fully controlled by the API caller via AgentRequest:
      - request.workflow  → which workflow to run (debate, research, ...)
      - request.tools     → which tools to inject (empty = all available)
    """

    # Names match agent_runner.REGISTRY for unified /tools discovery
    AVAILABLE_TOOLS = [
        "web_search",
        "scrape_url",
        "get_stock_price",
        "get_stock_history",
        "generate_stock_chart",
        "get_crypto_price",
        "generate_chart",
        "get_weather",
        "save_research_report",
        # Travel tools
        "search_flights",
        "search_hotels",
        "estimate_trip_budget",
        "search_places",
        "search_restaurants",
        "generate_itinerary",
        "get_local_transport_info",
        "get_distance_between_places",
        "generate_trip_summary",
        "get_currency_exchange",
        "get_geo_distance",
    ]

    # Map workflow name → handler method name
    WORKFLOW_REGISTRY = {
        "debate": "_execute_debate_workflow",
        "research": "_execute_research_workflow",
        "smart_assistant": "_execute_smart_assistant_workflow",
        "smart_travel_planner": "_execute_smart_travel_planner_workflow",
    }

    def __init__(self, model_client):
        self.model_client = model_client
        self._tool_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # IAgentOrchestrator interface
    # ------------------------------------------------------------------

    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Dispatch to the workflow specified in request.workflow."""
        workflow = request.workflow.lower()
        handler_name = self.WORKFLOW_REGISTRY.get(workflow)

        if not handler_name:
            available = list(self.WORKFLOW_REGISTRY.keys())
            return AgentResponse(
                answer=f"Unknown workflow '{workflow}'. Available: {available}",
                steps=[], tools_used=[], final_step=True
            )

        tools = self._resolve_tools(request.tools)
        handler = getattr(self, handler_name)

        try:
            return await handler(request.question, tools, request.max_steps)
        except Exception as e:
            logger.error("AutoGen workflow '%s' failed: %s", workflow, e, exc_info=True)
            return AgentResponse(answer=f"Workflow failed: {e}", steps=[], tools_used=[], final_step=True)

    def register_tool(self, tool: Any) -> None:
        pass  # Tool registration handled via request.tools

    def get_available_tools(self) -> List[str]:
        return self.AVAILABLE_TOOLS

    def get_available_workflows(self) -> List[str]:
        return list(self.WORKFLOW_REGISTRY.keys())

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def _resolve_tools(self, requested: List[str]) -> List[Callable]:
        """Return tool callables for the requested names (empty = all)."""
        registry = _register_tool_builders()
        names = requested if requested else list(registry.keys())
        tools = []
        for name in names:
            if name in registry:
                tools.append(registry[name])
            else:
                logger.warning("AutoGen: unknown tool '%s' requested, skipping", name)
        return tools

    def _resolve_agent_tools(self, names: List[str]) -> List[Callable]:
        registry = _register_tool_builders()
        agent_tools = []
        for name in names:
            func = registry.get(name)
            if not func:
                continue

            def make_tool(tool_name: str, tool_func: Callable) -> Callable:
                def agent_tool(**kwargs):
                    return tool_func(**kwargs)

                agent_tool.__name__ = tool_name
                agent_tool.__doc__ = inspect.getdoc(tool_func) or ""
                agent_tool.__signature__ = inspect.signature(tool_func)
                agent_tool.__annotations__ = getattr(tool_func, "__annotations__", {})
                return agent_tool

            agent_tools.append(make_tool(name, func))
        return agent_tools

    def _tool_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"

    def _get_cached_tool_result(self, key: str) -> Optional[Dict[str, Any]]:
        return self._tool_cache.get(key)

    def _cache_tool_result(self, key: str, result: Dict[str, Any]) -> None:
        self._tool_cache[key] = result

    def _build_tool_catalog(self, available_tool_names: List[str]) -> List[Dict[str, Any]]:
        registry = _register_tool_builders()
        catalog = []
        for name in available_tool_names:
            func = registry.get(name)
            if not func:
                continue
            signature = inspect.signature(func)
            parameters = []
            for param_name, param in signature.parameters.items():
                required = param.default is inspect.Parameter.empty
                parameters.append({
                    "name": param_name,
                    "required": required,
                    "default": None if required else param.default,
                })
            catalog.append({
                "name": name,
                "description": inspect.getdoc(func) or "",
                "parameters": parameters,
            })
        return catalog

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _fallback_tool_plan(self, query: str, available_tool_names: List[str]) -> Dict[str, Any]:
        tool_calls = []
        if "web_search" in available_tool_names:
            tool_calls.append({"name": "web_search", "args": {"query": query}})
        return {
            "intent": "GENERAL_QUERY",
            "confidence": 0.0,
            "tool_calls": tool_calls,
            "routing_source": "fallback",
        }

    def _normalize_tool_plan(
        self,
        raw_plan: Dict[str, Any],
        query: str,
        available_tool_names: List[str],
    ) -> Dict[str, Any]:
        registry = _register_tool_builders()
        intent = str(raw_plan.get("intent") or "GENERAL_QUERY").upper()
        try:
            confidence = float(raw_plan.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(confidence, 1.0))

        raw_tool_calls = raw_plan.get("tool_calls") or raw_plan.get("tools") or []
        normalized_calls = []
        for item in raw_tool_calls:
            if isinstance(item, str):
                name, args = item, {}
            elif isinstance(item, dict):
                name = item.get("name") or item.get("tool")
                args = item.get("args") or item.get("arguments") or {}
            else:
                continue

            if name not in available_tool_names or name not in registry:
                continue
            if not isinstance(args, dict):
                args = {}

            signature = inspect.signature(registry[name])
            allowed_args = {key: value for key, value in args.items() if key in signature.parameters}

            if name == "web_search":
                allowed_args.setdefault("query", query)
            elif name == "scrape_url" and not allowed_args.get("url"):
                url_match = re.search(r"https?://[^\s)>\]]+", query)
                if not url_match:
                    continue
                allowed_args["url"] = url_match.group(0).rstrip(".,")

            missing_required = [
                param_name
                for param_name, param in signature.parameters.items()
                if param.default is inspect.Parameter.empty and param_name not in allowed_args
            ]
            if missing_required:
                logger.warning(
                    "AutoGen router skipped tool '%s'; missing args=%s",
                    name, missing_required,
                )
                continue
            normalized_calls.append({"name": name, "args": allowed_args})

        if not normalized_calls:
            return self._fallback_tool_plan(query, available_tool_names)

        return {
            "intent": intent,
            "confidence": confidence,
            "tool_calls": normalized_calls,
            "routing_source": "llm",
        }

    async def _select_smart_assistant_tools(
        self,
        query: str,
        available_tool_names: List[str],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Agent 1 — Tool Selector: decides which tools to call and with what args.
        Max 2 steps. Returns (route_plan, selector_steps).
        """
        if not available_tool_names:
            return (
                {"intent": "GENERAL_QUERY", "confidence": 0.0, "tool_calls": [], "routing_source": "none"},
                [],
            )

        catalog = self._build_tool_catalog(available_tool_names)
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

        try:
            selector_result, selector_steps, _ = await self._run_team(selector_team, selector_task)
            parsed = self._extract_json_object(selector_result)
            if not parsed:
                raise ValueError(f"Selector did not return JSON: {selector_result!r}")
            route_plan = self._normalize_tool_plan(parsed, query, available_tool_names)
            return route_plan, selector_steps
        except Exception as exc:
            logger.warning("AutoGen selector failed; falling back: %s", exc, exc_info=True)
            return self._fallback_tool_plan(query, available_tool_names), []

    async def _execute_tool(self, tool_name: str, func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = self._tool_cache_key(tool_name, args)
        cached = self._get_cached_tool_result(cache_key)
        if cached is not None:
            return {"tool": tool_name, "args": args, "result": cached, "duration_ms": 0.0, "cached": True}

        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(func, **args)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if isinstance(result, dict):
            self._cache_tool_result(cache_key, result)
        return {"tool": tool_name, "args": args, "result": result, "duration_ms": duration_ms, "cached": False}

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        registry = _register_tool_builders()
        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            func = registry.get(tool_name)
            if not func:
                continue
            tasks.append(self._execute_tool(tool_name, func, tool_call.get("args", {})))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_smart_assistant_tools_with_agent(
        self,
        query: str,
        tool_calls: List[Dict[str, Any]],
        selected_tool_names: List[str],
    ) -> tuple[str, List[Dict[str, Any]], set]:
        """Agent 2 — Tool Executor: runs the selected tools and returns raw results."""
        selected_tools = self._resolve_agent_tools(selected_tool_names)
        if not selected_tools:
            return "[]", [], set()

        executor = AssistantAgent(
            name="ToolExecutor",
            system_message=(
                "You are a tool executor. Call each tool exactly as specified with the given arguments. "
                "Do not add or skip tools. After all calls complete, return a compact JSON array: "
                '[{"tool":"name","args":{},"result":{}}].'
            ),
            model_client=self.model_client,
            tools=selected_tools,
        )
        executor_team = RoundRobinGroupChat(
            participants=[executor],
            termination_condition=MaxMessageTermination(max_messages=max(3, len(tool_calls) + 2)),
        )
        executor_task = (
            f"User query: {query}\n\n"
            f"Execute these tool calls:\n{json.dumps(tool_calls, indent=2, default=str)}\n\n"
            "Return JSON results only."
        )
        return await self._run_team(executor_team, executor_task)

    def _get_research_tools(self, tools: List[Callable]) -> List[Callable]:
        """Return only data-gathering tools (excludes save_text_file)."""
        return [t for t in tools if t.__name__ != "save_text_file_tool"]

    def _get_save_tools(self, tools: List[Callable]) -> List[Callable]:
        """Return only file-saving tools."""
        return [t for t in tools if t.__name__ == "save_research_report_tool"]

    # ------------------------------------------------------------------
    # Shared stream runner
    # ------------------------------------------------------------------

    async def _run_team(
        self,
        team: RoundRobinGroupChat,
        task: str
    ) -> tuple[str, List[Dict[str, Any]], set]:

        steps = []
        tools_used = set()

        final_result = ""
        last_non_empty_message = ""

        step_index = 0

        async for message in team.run_stream(task=task):

            # -------------------------------------------------
            # Skip messages without content
            # -------------------------------------------------

            if not hasattr(message, "content"):
                continue

            content = message.content

            if content is None:
                continue

            content_str = str(content).strip()

            # -------------------------------------------------
            # Skip empty chunks
            # -------------------------------------------------

            if not content_str:
                continue

            step_index += 1

            step: Dict[str, Any] = {
                "step": step_index,
                "agent": getattr(message, "source", "unknown"),
                "content": content_str,
                "type": (
                    "tool_call"
                    if hasattr(message, "tool_calls")
                    and message.tool_calls
                    else "reasoning"
                ),
            }

            # -------------------------------------------------
            # Track tool calls
            # -------------------------------------------------

            if hasattr(message, "tool_calls") and message.tool_calls:

                step["tools_called"] = []

                for tc in message.tool_calls:

                    tool_name = (
                        tc.name
                        if hasattr(tc, "name")
                        else str(tc)
                    )

                    step["tools_called"].append(tool_name)

                    tools_used.add(tool_name)

            steps.append(step)

            # -------------------------------------------------
            # Save last valid response
            # -------------------------------------------------

            last_non_empty_message = content_str

        # -----------------------------------------------------
        # Final fallback safety
        # -----------------------------------------------------

        final_result = last_non_empty_message

        return final_result, steps, tools_used

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    async def _execute_debate_workflow(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        """Three-agent debate: Advocate vs Critic, moderated by Moderator."""
        advocate = AssistantAgent(
            name="Advocate",
            system_message="You argue FOR the given topic with strong supporting evidence. Be concise.",
            model_client=self.model_client,
            tools=tools or None,
        )
        critic = AssistantAgent(
            name="Critic",
            system_message="You argue AGAINST the given topic with counterarguments. Be concise.",
            model_client=self.model_client,
            tools=tools or None,
        )
        moderator = AssistantAgent(
            name="Moderator",
            system_message="Moderate the debate and provide a final balanced summary. Be concise.",
            model_client=self.model_client,
        )

        team = RoundRobinGroupChat(
            participants=[advocate, critic, moderator],
            termination_condition=MaxMessageTermination(max_messages=max_steps)
        )
        final_result, steps, tools_used = await self._run_team(team, f"Debate topic: {query}")

        return AgentResponse(
            answer=final_result,
            steps=steps,
            tools_used=list(tools_used) or [t.__name__ for t in tools],
            final_step=True
        )

    async def _execute_research_workflow(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        """Six-agent research pipeline: Plan → Research → Verify → Analyse → Evaluate → Report."""
        planner = AssistantAgent(
            name="Planner",
            system_message="Break research queries into structured tasks.",
            model_client=self.model_client,
        )
        researcher = AssistantAgent(
            name="Researcher",
            system_message="Gather factual evidence with citations only.",
            model_client=self.model_client,
            tools=self._get_research_tools(tools) or None,
        )
        verifier = AssistantAgent(
            name="Verifier",
            system_message="Verify sources, remove duplicates, check consistency.",
            model_client=self.model_client,
        )
        analyst = AssistantAgent(
            name="Analyst",
            system_message="Synthesize verified findings into insights.",
            model_client=self.model_client,
        )
        evaluator = AssistantAgent(
            name="Evaluator",
            system_message="Critique analysis for hallucinations, gaps, and weak evidence.",
            model_client=self.model_client,
        )
        report_writer = AssistantAgent(
            name="ReportWriter",
            system_message=(
                "Convert final analysis into a professional research report.\n"
                "Call save_research_report with:\n"
                "  title    = concise report title\n"
                "  query    = the original research question\n"
                "  summary  = 1-3 sentence executive summary\n"
                "  markdown = full report body in markdown (Key Findings, Evidence, Risks, Conclusion)\n"
                "  metadata = JSON string with tags and topic, e.g. '{\"topic\": \"AI\", \"tags\": [\"research\"]}'\n"
                "  sources  = newline-separated URLs or citations from Researcher"
            ),
            model_client=self.model_client,
            tools=self._get_save_tools(tools) or None,
        )

        team = RoundRobinGroupChat(
            participants=[planner, researcher, verifier, analyst, evaluator, report_writer],
            termination_condition=MaxMessageTermination(max_messages=max_steps)
        )

        task = (
            f"Research this topic thoroughly:\n\n{query}\n\n"
            "Final step: Save the final report using save_text_file tool."
        )
        final_result, steps, tools_used = await self._run_team(team, task)

        return AgentResponse(
            answer=final_result,
            steps=steps,
            tools_used=list(tools_used),
            final_step=True
        )

    async def _execute_smart_assistant_workflow(self, query: str, tools: List[Callable], max_steps: int) -> AgentResponse:
        """
        3-agent smart assistant pipeline:
          Agent 1 (ToolSelector)  — decides which tools to call (max 2 steps)
          Agent 2 (ToolExecutor)  — executes the selected tools
          Agent 3 (Summarizer)    — summarizes results (max = user-defined max_steps)
        """
        registry = _register_tool_builders()
        available_tool_names = [
            name for name, func in registry.items()
            if (not tools or func in tools) and name != "save_research_report"
        ]

        # ── Agent 1: Tool Selector ────────────────────────────────────────────
        route_plan, selector_steps = await self._select_smart_assistant_tools(query, available_tool_names)
        intent = route_plan["intent"]
        confidence = route_plan["confidence"]
        tool_calls = route_plan["tool_calls"]
        selected_tool_names = [tc["name"] for tc in tool_calls]

        if not tool_calls and "web_search" in available_tool_names:
            tool_calls = [{"name": "web_search", "args": {"query": query}}]
            selected_tool_names = ["web_search"]

        if not selector_steps:
            selector_steps = [{
                "step": 1,
                "agent": "ToolSelector",
                "content": json.dumps(
                    {"intent": intent, "confidence": confidence,
                     "routing_source": route_plan.get("routing_source"),
                     "tool_calls": tool_calls},
                    default=str,
                ),
                "type": "tool_routing",
            }]

        # ── Agent 2: Tool Executor ────────────────────────────────────────────
        executor_result, executor_steps, executor_tools_used = (
            await self._execute_smart_assistant_tools_with_agent(query, tool_calls, selected_tool_names)
        )

        # ── Agent 3: Summarizer ───────────────────────────────────────────────
        summarizer = AssistantAgent(
            name="Summarizer",
            system_message=(
                "You are the final assistant. Tool results are already provided — do not call any tools. "
                "Summarize the results clearly and concisely. "
                "When the answer contains multiple independent facts (e.g. weather + stock price), "
                "return a JSON object; otherwise return plain text."
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
            f"Tools used: {json.dumps(selected_tool_names)}\n"
            f"Tool results:\n{executor_result}"
        )
        final_result, summary_steps, summary_tools_used = await self._run_team(summarizer_team, summarizer_task)

        # ── Merge steps with sequential numbering ─────────────────────────────
        pre_summary = selector_steps + executor_steps
        for i, step in enumerate(pre_summary, start=1):
            step["step"] = i
        for step in summary_steps:
            step["step"] = step.get("step", 0) + len(pre_summary)

        tools_used = set(executor_tools_used) or set(selected_tool_names)
        tools_used.update(summary_tools_used)

        return AgentResponse(
            answer=final_result,
            steps=pre_summary + summary_steps,
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

    # ------------------------------------------------------------------
    # Smart Travel Planner workflow
    # ------------------------------------------------------------------

    # Supported travel intents and their tool mappings
    _TRAVEL_INTENT_KEYWORDS: List[tuple] = [
        ("FLIGHT_SEARCH",        ["flight", "fly", "airline", "airfare", "plane ticket"]),
        ("HOTEL_SEARCH",         ["hotel", "stay", "accommodation", "hostel", "resort", "lodge"]),
        ("WEATHER_TRAVEL",       ["weather", "climate", "temperature", "rain", "monsoon", "season", "cold", "warm"]),
        ("RESTAURANT_SEARCH",    ["restaurant", "food", "eat", "dining", "cuisine", "cafe"]),
        ("TRANSPORT_QUERY",      ["transport", "bus", "train", "cab", "taxi", "auto", "how to reach", "distance"]),
        ("LOCAL_ATTRACTIONS",    ["places", "attractions", "sightseeing", "tourist", "things to do", "explore", "visit", "beach"]),
        ("BUDGET_TRAVEL",        ["budget", "cheap", "affordable", "under", "cost", "expense", "\u20b9", "inr", "rupee", "$", "dollar", "ruble", "yuan"]),
        ("ITINERARY_PLANNING",   ["itinerary", "plan", "day", "schedule", "trip plan", "days"]),
        ("GENERAL_TRAVEL_QUERY", []),
    ]

    _TRAVEL_INTENT_TOOLS: Dict[str, List[str]] = {
        "FLIGHT_SEARCH":       ["search_flights", "get_geo_distance"],
        "HOTEL_SEARCH":        ["search_hotels", "get_weather"],
        "WEATHER_TRAVEL":      ["get_weather", "search_places"],
        "RESTAURANT_SEARCH":   ["search_restaurants"],
        "TRANSPORT_QUERY":     ["get_local_transport_info", "get_geo_distance"],
        "LOCAL_ATTRACTIONS":   ["search_places", "get_weather"],
        "BUDGET_TRAVEL":       ["estimate_trip_budget", "search_hotels", "search_flights", "get_weather", "get_currency_exchange"],
        "ITINERARY_PLANNING":  ["generate_itinerary", "search_places", "get_weather", "estimate_trip_budget", "get_geo_distance"],
        "GENERAL_TRAVEL_QUERY": ["generate_trip_summary", "get_weather", "search_places", "estimate_trip_budget"],
    }

    _TRAVEL_TOOL_NAMES: Set[str] = {
        "search_flights",
        "search_hotels",
        "estimate_trip_budget",
        "search_places",
        "search_restaurants",
        "generate_itinerary",
        "get_local_transport_info",
        "get_distance_between_places",
        "generate_trip_summary",
        "get_currency_exchange",
        "get_geo_distance",
        "get_weather",
        # TODO: Add web_search/scrape_url back later as an optional travel enrichment layer.
    }

    def _classify_travel_intent(self, query: str) -> str:
        """Classify travel intent via keyword matching."""
        q = query.lower()
        for intent, keywords in self._TRAVEL_INTENT_KEYWORDS:
            if not keywords:
                return intent
            if any(kw in q for kw in keywords):
                return intent
        return "GENERAL_TRAVEL_QUERY"

    def _extract_travel_entities(self, query: str) -> Dict[str, Any]:
        """Extract destination, days, budget, source, travelers, currency, preferences from query."""
        q = query.lower()
        
        # Expanded destination database with regions
        known_places = [
            # India
            "goa", "jaipur", "kerala", "mumbai", "delhi", "bangalore", "hyderabad",
            "manali", "shimla", "udaipur", "agra", "varanasi", "kolkata", "chennai",
            "kathmandu", "katmandu", "rishikesh", "darjeeling", "ooty", "coorg",
            # Middle East
            "dubai", "abu dhabi", "doha", "riyadh", "muscat", "bahrain", "kuwait",
            # Europe
            "paris", "london", "rome", "italy", "venice", "milan", "barcelona", "madrid",
            "berlin", "amsterdam", "prague", "vienna", "budapest", "moscow",
            # Asia
            "singapore", "bangkok", "bali", "phuket", "tokyo", "seoul", "beijing",
            "hong kong", "kuala lumpur", "hanoi", "ho chi minh",
            # Americas
            "new york", "los angeles", "san francisco", "miami", "chicago",
            "toronto", "vancouver", "mexico city", "cancun",
        ]
        
        # Extract destination (check multi-word first)
        destination = None
        for place in sorted(known_places, key=len, reverse=True):
            if place in q:
                destination = place.title()
                break
        
        # Extract source/origin
        from_patterns = [
            r"from\s+([a-z]+(?:\s+[a-z]+)?)",
            r"coming\s+from\s+([a-z]+(?:\s+[a-z]+)?)",
            r"traveling\s+from\s+([a-z]+(?:\s+[a-z]+)?)",
        ]
        source = None
        for pattern in from_patterns:
            from_m = re.search(pattern, q)
            if from_m:
                potential_source = from_m.group(1).strip()
                # Check if it's a known place
                for place in known_places:
                    if place in potential_source:
                        source = place.title()
                        break
                if source:
                    break
        
        # Detect origin from country/region mentions if no explicit "from"
        if not source:
            if any(word in q for word in ["russia", "russian", "moscow"]):
                source = "Moscow"
            elif any(word in q for word in ["china", "chinese", "beijing"]):
                source = "Beijing"
            elif any(word in q for word in ["usa", "america", "american"]):
                source = "New York"
        
        # If still no source and destination exists, use a default
        if not source and destination:
            source = "Delhi"  # Default for Indian destinations
        
        # Extract days
        days_m = re.search(r"(\d+)\s*(?:day|days|night|nights)", q)
        days = int(days_m.group(1)) if days_m else 3
        
        # Extract budget with multi-currency support
        budget = None
        budget_currency = "INR"
        
        # Try different currency patterns
        currency_patterns = [
            (r"(?:under|below|within|budget|\$)\s*(\d[\d,]*)", "USD"),
            (r"(?:under|below|within|budget|\u20b9|rs\.?|inr)\s*(\d[\d,]*)", "INR"),
            (r"(?:under|below|within|budget|\u20bd|rub|ruble)\s*(\d[\d,]*)", "RUB"),
            (r"(?:under|below|within|budget|\u00a5|yuan|cny|rmb)\s*(\d[\d,]*)", "CNY"),
            (r"(?:under|below|within|budget|\u20ac|eur|euro)\s*(\d[\d,]*)", "EUR"),
        ]
        
        for pattern, currency in currency_patterns:
            budget_m = re.search(pattern, q)
            if budget_m:
                budget = int(budget_m.group(1).replace(",", ""))
                budget_currency = currency
                break
        
        # Extract travelers
        travelers_m = re.search(r"(\d+)\s*(?:person|people|traveler|travellers|pax)", q)
        travelers = int(travelers_m.group(1)) if travelers_m else 1
        
        # Extract preferences (beach, cold, family, etc.)
        preferences = []
        if any(word in q for word in ["beach", "beaches", "coastal", "sea", "ocean"]):
            preferences.append("beach")
        if any(word in q for word in ["cold", "snow", "winter", "skiing"]):
            preferences.append("cold_weather")
        if any(word in q for word in ["hot", "warm", "summer", "sunny"]):
            preferences.append("warm_weather")
        if any(word in q for word in ["family", "families", "kids", "children"]):
            preferences.append("family_friendly")
        if any(word in q for word in ["adventure", "trekking", "hiking", "sports"]):
            preferences.append("adventure")
        if any(word in q for word in ["luxury", "premium", "5-star", "upscale"]):
            preferences.append("luxury")
        if any(word in q for word in ["budget", "cheap", "affordable", "backpack"]):
            preferences.append("budget_travel")
        
        return {
            "destination": destination or "the destination",
            "days": days,
            "budget": budget,
            "budget_currency": budget_currency,
            "source": source,
            "travelers": travelers,
            "preferences": preferences,
        }

    def _select_travel_tools(self, query: str) -> tuple[str, List[str]]:
        """Return (intent, tool_names) for the query."""
        intent = self._classify_travel_intent(query)
        tools = self._TRAVEL_INTENT_TOOLS.get(intent, ["generate_trip_summary", "get_weather", "search_places"])
        tools = [name for name in tools if name in self._TRAVEL_TOOL_NAMES]
        logger.debug("TRAVEL_PLANNER: intent=%s | tools=%s", intent, tools)
        return intent, tools

    def _normalize_travel_entities(self, raw_entities: Any, query: str) -> Dict[str, Any]:
        entities = self._extract_travel_entities(query)
        if not isinstance(raw_entities, dict):
            return entities

        for key in ["destination", "source", "budget_currency"]:
            value = raw_entities.get(key)
            if isinstance(value, str) and value.strip():
                entities[key] = value.strip().title() if key != "budget_currency" else value.strip().upper()

        for key in ["days", "budget", "travelers"]:
            value = raw_entities.get(key)
            if value in (None, ""):
                continue
            try:
                entities[key] = int(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue

        preferences = raw_entities.get("preferences")
        if isinstance(preferences, list):
            entities["preferences"] = [str(pref).strip() for pref in preferences if str(pref).strip()]
        elif isinstance(preferences, str) and preferences.strip():
            entities["preferences"] = [pref.strip() for pref in preferences.split(",") if pref.strip()]

        return entities

    def _travel_args_for_tool(self, tool_name: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        dest = entities["destination"]
        days = str(entities["days"])
        budget = entities.get("budget")
        budget_str = str(budget) if budget else ""
        source = entities.get("source") or "Delhi"
        travelers = str(entities["travelers"])
        preferences = entities.get("preferences", [])

        if tool_name == "search_flights":
            return {"origin": source, "destination": dest, "budget": budget_str}
        if tool_name == "search_hotels":
            return {"destination": dest, "budget": budget_str, "days": days}
        if tool_name == "estimate_trip_budget":
            return {"destination": dest, "days": days, "travelers": travelers}
        if tool_name == "search_places":
            category = "beach" if "beach" in preferences else "tourist"
            return {"destination": dest, "category": category}
        if tool_name == "search_restaurants":
            return {"destination": dest}
        if tool_name == "generate_itinerary":
            return {"destination": dest, "days": days, "budget": budget_str}
        if tool_name == "get_local_transport_info":
            return {"destination": dest}
        if tool_name in {"get_distance_between_places", "get_geo_distance"}:
            return {"origin": source, "destination": dest}
        if tool_name == "generate_trip_summary":
            return {"destination": dest, "days": days, "budget": budget_str}
        if tool_name == "get_weather":
            return {"city": dest}
        if tool_name == "get_currency_exchange":
            return {
                "from_currency": entities.get("budget_currency", "INR"),
                "to_currency": "INR",
                "amount": float(budget or 1),
            }
        return {"query": query}

    def _fallback_travel_plan(self, query: str, available_tool_names: List[str]) -> Dict[str, Any]:
        intent, selected_tool_names = self._select_travel_tools(query)
        selected_tool_names = [name for name in selected_tool_names if name in available_tool_names]
        if not selected_tool_names and "generate_trip_summary" in available_tool_names:
            selected_tool_names = ["generate_trip_summary"]

        entities = self._extract_travel_entities(query)
        tool_calls = [
            {"name": name, "args": self._travel_args_for_tool(name, entities, query)}
            for name in selected_tool_names
        ]

        return {
            "intent": intent,
            "confidence": 0.70,
            "entities": entities,
            "tool_calls": tool_calls,
            "routing_source": "fallback",
        }

    def _normalize_travel_tool_plan(
        self,
        raw_plan: Dict[str, Any],
        query: str,
        available_tool_names: List[str],
    ) -> Dict[str, Any]:
        registry = _register_tool_builders()
        entities = self._normalize_travel_entities(raw_plan.get("entities"), query)
        intent = str(raw_plan.get("intent") or "GENERAL_TRAVEL_QUERY").upper()
        try:
            confidence = float(raw_plan.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(confidence, 1.0))

        raw_tool_calls = raw_plan.get("tool_calls") or raw_plan.get("tools") or []
        normalized_calls = []
        for item in raw_tool_calls:
            if isinstance(item, str):
                name, args = item, {}
            elif isinstance(item, dict):
                name = item.get("name") or item.get("tool")
                args = item.get("args") or item.get("arguments") or {}
            else:
                continue

            if name not in available_tool_names or name not in self._TRAVEL_TOOL_NAMES or name not in registry:
                continue
            if not isinstance(args, dict):
                args = {}

            signature = inspect.signature(registry[name])
            default_args = self._travel_args_for_tool(name, entities, query)
            allowed_args = {key: value for key, value in args.items() if key in signature.parameters}
            for key, value in default_args.items():
                allowed_args.setdefault(key, value)

            missing_required = [
                param_name
                for param_name, param in signature.parameters.items()
                if param.default is inspect.Parameter.empty and param_name not in allowed_args
            ]
            if missing_required:
                logger.warning(
                    "AutoGen travel router skipped tool '%s'; missing args=%s",
                    name, missing_required,
                )
                continue
            normalized_calls.append({"name": name, "args": allowed_args})

        if not normalized_calls:
            return self._fallback_travel_plan(query, available_tool_names)

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "tool_calls": normalized_calls,
            "routing_source": "llm",
        }

    async def _plan_smart_travel_tools(self, query: str, available_tool_names: List[str]) -> Dict[str, Any]:
        if not available_tool_names:
            return {
                "intent": "GENERAL_TRAVEL_QUERY",
                "confidence": 0.0,
                "entities": self._extract_travel_entities(query),
                "tool_calls": [],
                "routing_source": "none",
            }

        catalog = self._build_tool_catalog(available_tool_names)
        prompt = (
            "You are a tool router for a smart travel planner. Extract travel entities, identify intent, "
            "and choose the smallest useful set of travel tools with exact arguments.\n\n"
            "Return only valid JSON with this shape:\n"
            "{"
            "\"intent\":\"SHORT_TRAVEL_INTENT\","
            "\"confidence\":0.0,"
            "\"entities\":{\"destination\":\"Goa\",\"source\":\"Delhi\",\"days\":3,\"budget\":20000,"
            "\"budget_currency\":\"INR\",\"travelers\":1,\"preferences\":[\"beach\"]},"
            "\"tool_calls\":[{\"name\":\"tool_name\",\"args\":{\"arg\":\"value\"}}]"
            "}\n\n"
            "Rules:\n"
            "- Use only tools from the provided catalog.\n"
            "- Do not use web_search or scrape_url. Web enrichment will be added later.\n"
            "- Prefer travel-specific tools such as search_places, search_hotels, generate_itinerary, "
            "estimate_trip_budget, get_weather, and transport/distance tools.\n"
            "- If the user gives a non-INR budget, include get_currency_exchange.\n"
            "- If origin is not specified, use a reasonable source only when needed by a selected tool.\n\n"
            f"Tool catalog:\n{json.dumps(catalog, indent=2, default=str)}\n\n"
            f"User query: {query}"
        )

        try:
            from autogen_core.models import UserMessage

            response = await self.model_client.create([
                UserMessage(content=prompt, source="user")
            ])
            content = response.content if hasattr(response, "content") else str(response)
            parsed = self._extract_json_object(str(content))
            if not parsed:
                raise ValueError(f"Travel router did not return JSON: {content!r}")
            return self._normalize_travel_tool_plan(parsed, query, available_tool_names)
        except Exception as exc:
            logger.warning("AutoGen travel LLM router failed; falling back to keyword routing: %s", exc, exc_info=True)
            return self._fallback_travel_plan(query, available_tool_names)

    async def _execute_travel_tools_parallel(
        self, tool_names: List[str], entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute travel tools in parallel using asyncio.gather."""
        registry = _register_tool_builders()
        dest = entities["destination"]
        days = str(entities["days"])
        budget = entities.get("budget")
        budget_currency = entities.get("budget_currency", "INR")
        source = entities.get("source")
        travelers = str(entities["travelers"])
        preferences = entities.get("preferences", [])

        # Convert budget to INR if needed
        budget_inr = None
        currency_conversion_result = None
        if budget and budget_currency != "INR":
            conv_result = await self._execute_tool(
                "get_currency_exchange",
                registry["get_currency_exchange"],
                {"from_currency": budget_currency, "to_currency": "INR", "amount": float(budget)}
            )
            if conv_result["result"].get("status") == "success":
                budget_inr = int(conv_result["result"]["converted_amount"])
                currency_conversion_result = conv_result
        elif budget:
            budget_inr = budget

        budget_str = str(budget_inr) if budget_inr else ""

        # Build (tool_name, func, args) triples
        tool_calls = []
        for name in tool_names:
            if name not in registry:
                continue
            if name == "search_flights":
                args = {"origin": source or "Delhi", "destination": dest, "budget": budget_str}
            elif name == "search_hotels":
                args = {"destination": dest, "budget": budget_str, "days": days}
            elif name == "estimate_trip_budget":
                args = {"destination": dest, "days": days, "travelers": travelers}
            elif name == "search_places":
                category = "beach" if "beach" in preferences else "tourist"
                args = {"destination": dest, "category": category}
            elif name == "search_restaurants":
                args = {"destination": dest}
            elif name == "generate_itinerary":
                args = {"destination": dest, "days": days, "budget": budget_str}
            elif name == "get_local_transport_info":
                args = {"destination": dest}
            elif name == "get_distance_between_places":
                args = {"origin": source or "Delhi", "destination": dest}
            elif name == "get_geo_distance":
                args = {"origin": source or "Delhi", "destination": dest}
            elif name == "generate_trip_summary":
                args = {"destination": dest, "days": days, "budget": budget_str}
            elif name == "get_weather":
                args = {"city": dest}
            elif name == "get_currency_exchange":
                args = {
                    "from_currency": budget_currency,
                    "to_currency": "INR",
                    "amount": float(budget or 1),
                }
            else:
                args = {"destination": dest}
            tool_calls.append((name, registry[name], args))

        tasks = [self._execute_tool(name, fn, args) for name, fn, args in tool_calls]
        results = list(await asyncio.gather(*tasks))
        
        # Prepend currency conversion if it happened
        if currency_conversion_result:
            results.insert(0, currency_conversion_result)
        
        return results

    async def _execute_smart_travel_planner_workflow(
        self, query: str, tools: List[Callable], max_steps: int
    ) -> AgentResponse:
        """Smart travel planner: classify intent → extract entities → run tools in
        parallel → hand structured results to an AssistantAgent for a final plan.
        """
        registry = _register_tool_builders()
        if tools:
            available_tool_names = [
                name for name, func in registry.items()
                if func in tools and name in self._TRAVEL_TOOL_NAMES
            ]
        else:
            available_tool_names = [
                name for name in self._TRAVEL_TOOL_NAMES
                if name in registry
            ]

        route_plan = await self._plan_smart_travel_tools(query, available_tool_names)
        intent = route_plan["intent"]
        confidence = route_plan["confidence"]
        entities = route_plan["entities"]
        tool_calls = route_plan["tool_calls"]
        selected_tool_names = [tool_call["name"] for tool_call in tool_calls]

        if not tool_calls and "generate_trip_summary" in available_tool_names:
            tool_calls = [{
                "name": "generate_trip_summary",
                "args": self._travel_args_for_tool("generate_trip_summary", entities, query),
            }]
            selected_tool_names = ["generate_trip_summary"]

        logger.info(
            "TRAVEL_PLANNER: intent=%s | dest=%s | days=%s | budget=%s | tools=%s",
            intent, entities["destination"], entities["days"], entities.get("budget"), selected_tool_names,
        )

        # Web scraping/search is intentionally excluded for now; add it back later as
        # an optional enrichment step after the travel-specific tools are stable.
        tool_results = await self._execute_tool_calls(tool_calls)

        # Build step records for the response
        tool_steps = [
            {
                "step": i + 1,
                "tool": r["tool"],
                "args": r["args"],
                "output": r["result"],
                "duration_ms": r["duration_ms"],
                "cached": r["cached"],
            }
            for i, r in enumerate(tool_results)
        ]

        # Step 5: LLM aggregation — structured travel plan
        budget_inr = None
        for result in tool_results:
            if result["tool"] != "get_currency_exchange":
                continue
            conversion = result["result"]
            if isinstance(conversion, dict) and conversion.get("status") == "success":
                converted_amount = conversion.get("converted_amount")
                if converted_amount:
                    budget_inr = int(converted_amount)
                    break

        budget_display = ""
        if entities.get("budget"):
            if entities.get("budget_currency") == "INR":
                budget_display = f"₹{entities['budget']:,}"
            else:
                budget_display = f"{entities['budget']:,} {entities['budget_currency']}"
                if budget_inr:
                    budget_display += f" (≈₹{budget_inr:,})"
        else:
            budget_display = "flexible"
        
        preferences_str = ", ".join(entities.get("preferences", [])) or "general travel"
        
        planner_prompt = (
            f"You are an expert travel planner. The user asked: '{query}'\n\n"
            f"Detected intent: {intent}\n"
            f"Origin: {entities.get('source') or 'Not specified'} → Destination: {entities['destination']}\n"
            f"Duration: {entities['days']} days | Budget: {budget_display} | Travelers: {entities['travelers']}\n"
            f"Preferences: {preferences_str}\n\n"
            "Tool results (use ONLY these — do not hallucinate):\n"
            f"{json.dumps([r['result'] for r in tool_results], indent=2, default=str)}\n\n"
            "Generate a structured travel plan with:\n"
            "- Destination overview\n"
            "- Distance & travel time from origin (if available)\n"
            "- Weather summary (if available)\n"
            "- Budget breakdown (converted to local currency if needed)\n"
            "- Recommended hotels matching budget\n"
            "- Day-wise itinerary aligned with preferences\n"
            "- Local transport options\n"
            "- Top attractions / restaurants\n"
            "- 3-5 practical travel tips\n"
            "Be concise, practical, and tailor recommendations to user preferences."
        )

        planner_agent = AssistantAgent(
            name="TravelPlanner",
            system_message=(
                "You are a structured travel planning assistant. "
                "Use only the provided tool results to build a clear, day-wise travel plan. "
                "Do not call any additional tools."
            ),
            model_client=self.model_client,
        )

        team = RoundRobinGroupChat(
            participants=[planner_agent],
            termination_condition=MaxMessageTermination(max_messages=max_steps),
        )

        final_result, summary_steps, _ = await self._run_team(team, planner_prompt)
        all_steps = tool_steps + summary_steps
        tools_used = [r["tool"] for r in tool_results]

        return AgentResponse(
            answer=final_result,
            steps=all_steps,
            tools_used=tools_used,
            final_step=True,
            debug_info={
                "intent": intent,
                "confidence": confidence,
                "origin": entities.get("source"),
                "destination": entities["destination"],
                "days": entities["days"],
                "budget": entities.get("budget"),
                "budget_currency": entities.get("budget_currency"),
                "travelers": entities["travelers"],
                "preferences": entities.get("preferences", []),
                "selected_tools": selected_tool_names,
                "routing_source": route_plan.get("routing_source"),
                "tool_calls": tool_calls,
            },
        )
