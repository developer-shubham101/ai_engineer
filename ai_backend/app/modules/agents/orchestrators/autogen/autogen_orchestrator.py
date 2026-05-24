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

    def _extract_json_object(
            self,
            text: str
    ) -> Optional[Dict[str, Any]]:

        if not text:
            return None

        text = str(text).strip()

        # -------------------------------------------------
        # FAST PATH
        # If response itself is already pure JSON
        # -------------------------------------------------

        try:
            parsed = json.loads(text)
            return parsed

        except Exception:
            pass

        # -------------------------------------------------
        # Extract from markdown ```json blocks
        # -------------------------------------------------

        json_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )

        if json_block_match:
            candidate = json_block_match.group(1).strip()

            try:
                parsed = json.loads(candidate)

                if isinstance(parsed, dict):
                    return parsed

            except Exception as e:
                logger.warning(
                    "Markdown JSON parse failed: %s",
                    e,
                )

        # -------------------------------------------------
        # Generic object extraction fallback
        # -------------------------------------------------

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if not match:
            return None

        candidate = match.group(0).strip()

        # -------------------------------------------------
        # First normal parse attempt
        # -------------------------------------------------

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except Exception as e:

            logger.warning(
                "JSON parse failed: %s",
                e,
            )

        # -------------------------------------------------
        # Auto-repair malformed JSON
        # Handles:
        # - extra closing braces
        # - trailing garbage
        # - accidental tokens at end
        # -------------------------------------------------

        repaired = candidate

        while repaired:

            try:
                parsed = json.loads(repaired)

                if isinstance(parsed, dict):
                    logger.warning(
                        "JSON auto-repair succeeded"
                    )

                    return parsed

            except Exception:
                pass

            repaired = repaired[:-1].strip()

        # -------------------------------------------------
        # Failed completely
        # -------------------------------------------------

        logger.warning(
            "Unable to extract valid JSON from selector output"
        )

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
            allowed_args = {
                key: value
                for key, value in args.items()
                if key in signature.parameters
            }

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

            logger.warning("Parsed: %s", parsed, exc_info=True)

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
        # -------------------------------------------------
        # DETERMINISTIC TOOL EXECUTION
        # -------------------------------------------------

        tool_results = await self._execute_tool_calls(
            tool_calls
        )

        executor_steps = []

        for idx, result in enumerate(tool_results, start=1):

            executor_steps.append({
                "step": idx,
                "agent": "ToolExecutor",
                "type": "tool_execution",
                "tool": result["tool"],
                "args": result["args"],
                "content": json.dumps(
                    result["result"],
                    indent=2,
                    default=str
                ),
                "duration_ms": result.get("duration_ms"),
                "cached": result.get("cached"),
            })

        executor_tools_used = {
            result["tool"]
            for result in tool_results
        }

        executor_result = json.dumps(
            tool_results,
            indent=2,
            default=str
        )

        # ── Agent 3: Summarizer ───────────────────────────────────────────────
        summarizer = AssistantAgent(
            name="Summarizer",
            system_message=(
                "You are the final assistant. Tool results are already provided — do not call any tools. "
                "Summarize the results clearly and concisely. "
                "When the answer contains multiple independent facts (e.g. weather + stock price), "
                "return plain text always in fomated way so user can ready."
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
 
 

    def _normalize_travel_tool_plan(
        self,
        raw_plan: Dict[str, Any],
        query: str,
        available_tool_names: List[str],
    ) -> Dict[str, Any]:
        registry = _register_tool_builders()
         
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
             
            allowed_args = {
                key: value
                for key, value in args.items()
                if key in signature.parameters
            }
             

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
        
        entities = raw_plan.get("entities") or {}

        return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "tool_calls": normalized_calls,
        "routing_source": "llm",
    }

    def _coerce_tool_args(
        self,
        func: Callable,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:

        signature = inspect.signature(func)
        normalized = {}

        for name, value in args.items():

            param = signature.parameters.get(name)

            if not param:
                continue

            annotation = param.annotation

            try:
                if annotation == int:
                    value = int(value)

                elif annotation == float:
                    value = float(value)

                elif annotation == str:
                    value = str(value)

            except Exception:
                pass

            normalized[name] = value

        return normalized
    def _fallback_travel_plan(
        self,
        query: str,
        available_tool_names: List[str],
    ) -> Dict[str, Any]:

        tool_calls = []

        if "generate_trip_summary" in available_tool_names:
            tool_calls.append({
                "name": "generate_trip_summary",
                "args": {
                    "destination": query,
                    "days": "3",
                    "budget": "",
                },
            })

        return {
            "intent": "GENERAL_TRAVEL_QUERY",
            "confidence": 0.0,
            "entities": {},
            "tool_calls": tool_calls,
            "routing_source": "fallback",
        }
    
    
    async def _select_smart_travel_planner_tools(
        self,
        query: str,
        available_tool_names: List[str],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        AI-based travel tool selector.

        AI decides:
        - intent
        - entities
        - tools
        - tool arguments

        No keyword routing.
        No hardcoded intent mapping.
        """

        if not available_tool_names:
            return (
                {
                    "intent": "GENERAL_TRAVEL_QUERY",
                    "confidence": 0.0,
                    "entities": {},
                    "tool_calls": [],
                    "routing_source": "none",
                },
                [],
            )

        catalog = self._build_tool_catalog(available_tool_names)

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
            model_client=self.model_client,
        )

        selector_team = RoundRobinGroupChat(
            participants=[selector],
            termination_condition=MaxMessageTermination(max_messages=2),
        )

        selector_task = (
            f"Available tools:\n"
            f"{json.dumps(catalog, indent=2, default=str)}\n\n"
            f"User query:\n{query}\n\n"
            "Return ONLY JSON."
        )

        try:
            selector_result, selector_steps, _ = await self._run_team(
                selector_team,
                selector_task,
            )

            parsed = self._extract_json_object(selector_result)

            if not parsed:
                raise ValueError(
                    f"Travel selector did not return valid JSON: {selector_result!r}"
                )

            route_plan = self._normalize_travel_tool_plan(
                parsed,
                query,
                available_tool_names,
            )

            return route_plan, selector_steps

        except Exception as exc:
            logger.warning(
                "Travel selector failed, fallback triggered: %s",
                exc,
                exc_info=True,
            )

            return (
                self._fallback_travel_plan(
                    query,
                    available_tool_names,
                ),
                [],
            )

    async def _execute_smart_travel_planner_workflow(
        self, query: str, tools: List[Callable], max_steps: int
    ) -> AgentResponse:
        """
        3-agent travel planner pipeline:
          Agent 1 (TravelToolSelector) - decides which travel tools to call (max 2 steps)
          Agent 2 (ToolExecutor)       - executes the selected tools
          Agent 3 (TravelPlanner)      - summarizes into a travel plan (max = user-defined max_steps)
        """
        registry = _register_tool_builders()
        available_tool_names = [
            name for name, func in registry.items()
            if (not tools or func in tools) and name in self._TRAVEL_TOOL_NAMES
        ]

        # -- Agent 1: Travel Tool Selector ------------------------------------
        route_plan, selector_steps = await self._select_smart_travel_planner_tools(query, available_tool_names)
        intent = route_plan["intent"]
        confidence = route_plan["confidence"]
        entities = route_plan["entities"]
        tool_calls = route_plan["tool_calls"]
        selected_tool_names = [tc["name"] for tc in tool_calls]

 

        if not selector_steps:
            selector_steps = [{
                "step": 1,
                "agent": "TravelToolSelector",
                "content": json.dumps(
                    {"intent": intent, "confidence": confidence, "entities": entities,
                     "routing_source": route_plan.get("routing_source"),
                     "tool_calls": tool_calls},
                    default=str,
                ),
                "type": "tool_routing",
            }]

        logger.info(
            "TRAVEL_PLANNER: intent=%s | dest=%s | days=%s | budget=%s | tools=%s",
            intent, entities.get("destination"), entities.get("days"), entities.get("budget"), selected_tool_names,
        )

        # -- Agent 2: Tool Executor -------------------------------------------
        tool_results = await self._execute_tool_calls(tool_calls)

        if not tool_results:
            return AgentResponse(
                answer="Unable to gather travel information right now.",
                steps=selector_steps,
                tools_used=[],
                final_step=True,
                debug_info={
                    "intent": intent,
                    "confidence": confidence,
                    "routing_source": route_plan.get("routing_source"),
                },
            )

        executor_steps = [
            {
                "step": idx,
                "agent": "ToolExecutor",
                "type": "tool_execution",
                "tool": result["tool"],
                "args": result["args"],
                "content": json.dumps(result["result"], indent=2, default=str),
                "duration_ms": result.get("duration_ms"),
                "cached": result.get("cached"),
            }
            for idx, result in enumerate(tool_results, start=1)
        ]
        executor_tools_used = {result["tool"] for result in tool_results}
        executor_result = json.dumps(tool_results, indent=2, default=str)

        # -- Agent 3: Travel Planner (Summarizer) -----------------------------
        preferences = entities.get("preferences") or []

        if not isinstance(preferences, list):
            preferences = [str(preferences)]

        preferences_str = ", ".join(preferences) or "general travel"

        summarizer = AssistantAgent(
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
                "- Overview\n"
                "- Budget\n"
                "- Hotels\n"
                "- Attractions\n"
                "- Weather\n"
                "- Transport\n"
                "- Tips\n\n"

                "If some information is unavailable, skip that section.\n"
                "Be concise but practical."
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
            f"Origin: {entities.get('source') or 'Not specified'} | Destination: {entities.get('destination')}\n"
            f"Duration: {entities.get('days')} days | Travelers: {entities.get('travelers')}\n"
            f"Preferences: {preferences_str}\n"
            f"Tools used: {json.dumps(selected_tool_names)}\n"
            f"Tool results:\n{executor_result}"
        )
        final_result, summary_steps, summary_tools_used = await self._run_team(summarizer_team, summarizer_task)

        # -- Merge steps with sequential numbering ----------------------------
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
