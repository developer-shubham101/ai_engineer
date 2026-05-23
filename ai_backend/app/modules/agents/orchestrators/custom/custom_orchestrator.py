"""Agent orchestrator with safety constraints."""

import logging
import time
from typing import Dict, Any, List, Optional

from ...interfaces import IAgentOrchestrator, ITool, AgentRequest, AgentResponse
from ...utils import StepFormatter

logger = logging.getLogger(__name__)


class CustomOrchestrator(IAgentOrchestrator):
    """Agent orchestrator with safety constraints."""

    def __init__(self, max_steps: int = 5):
        self.tools: Dict[str, ITool] = {}
        self.max_steps = max_steps

    def register_tool(self, tool: ITool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.debug("ORCHESTRATOR: registered tool=%s", tool.name)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process agent request with safety constraints."""
        start_time = time.time()
        steps = []
        tools_used = []
        user_id = (user or {}).get("user_id", "anonymous")

        logger.info(
            "ORCHESTRATOR_START: user=%s | question_len=%d | tools_requested=%s | max_steps=%d",
            user_id, len(request.question), request.tools or "all", request.max_steps
        )
        logger.debug("ORCHESTRATOR_START: question=%r", request.question[:120])

        try:
            max_steps = min(request.max_steps, self.max_steps)
            available_tools = self.get_available_tools()
            enabled_tools = [t for t in request.tools if t in available_tools] if request.tools else available_tools

            logger.debug(
                "ORCHESTRATOR: available_tools=%s | enabled_tools=%s | effective_max_steps=%d",
                available_tools, enabled_tools, max_steps
            )

            if not enabled_tools:
                logger.warning("ORCHESTRATOR: no valid tools available | requested=%s | available=%s",
                               request.tools, available_tools)
                return AgentResponse(
                    answer="No valid tools available for this request.",
                    steps=[],
                    tools_used=[],
                    debug_info={"error": "No valid tools"}
                )

            answer = await self._simulate_agent_workflow(
                question=request.question,
                enabled_tools=enabled_tools,
                max_steps=max_steps,
                user=user,
                steps=steps,
                tools_used=tools_used
            )

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                "ORCHESTRATOR_DONE: user=%s | steps=%d | tools_used=%s | answer_len=%d | duration_ms=%d",
                user_id, len(steps), list(set(tools_used)), len(answer), processing_time
            )

            debug_info = None
            if request.debug:
                debug_info = {
                    "processing_time_ms": processing_time,
                    "available_tools": available_tools,
                    "enabled_tools": enabled_tools,
                    "max_steps": max_steps,
                    "actual_steps": len(steps)
                }
                logger.debug("ORCHESTRATOR_DEBUG: %s", debug_info)

            return AgentResponse(
                answer=answer,
                steps=steps,
                tools_used=list(set(tools_used)),
                debug_info=debug_info
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(
                "ORCHESTRATOR_ERROR: user=%s | error=%s | duration_ms=%d",
                user_id, e, processing_time, exc_info=True
            )
            return AgentResponse(
                answer=f"Agent processing failed: {str(e)}",
                steps=steps,
                tools_used=tools_used,
                debug_info={"error": str(e)}
            )

    async def _simulate_agent_workflow(
        self,
        question: str,
        enabled_tools: List[str],
        max_steps: int,
        user: Optional[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        tools_used: List[str]
    ) -> str:
        """Simulate agent workflow with tool selection logic."""
        context = {"user": user or {}, "question": question}
        q = question.lower()

        logger.debug("WORKFLOW: routing question | keywords detected from: %r", question[:80])

        if "ticket" in q and "get_user_tickets" in enabled_tools:
            logger.debug("WORKFLOW: route=ticket_flow")
            result = await self._execute_tool("get_user_tickets", "current", context, steps, tools_used)
            if "TKT-" in result and "get_ticket_comments" in enabled_tools and len(steps) < max_steps:
                logger.debug("WORKFLOW: ticket found, fetching comments")
                await self._execute_tool("get_ticket_comments", "TKT-001", context, steps, tools_used)

        elif "search" in q or "document" in q:
            logger.debug("WORKFLOW: route=document_search")
            if "search_documents" in enabled_tools:
                await self._execute_tool("search_documents", question, context, steps, tools_used)

        elif any(kw in q for kw in ["web", "internet", "latest", "current", "news"]):
            logger.debug("WORKFLOW: route=web_search")
            if "web_search" in enabled_tools:
                result = await self._execute_tool("web_search", question, context, steps, tools_used)
                if "scrape_url" in enabled_tools and len(steps) < max_steps:
                    import re
                    urls = re.findall(r'https?://[^\s]+', result)
                    if urls:
                        logger.debug("WORKFLOW: auto-scraping first URL=%s", urls[0][:80])
                        await self._execute_tool("scrape_url", urls[0], context, steps, tools_used)

        elif "analyze" in q or "data" in q:
            logger.debug("WORKFLOW: route=data_analysis")
            if "analyze_data" in enabled_tools:
                await self._execute_tool("analyze_data", question, context, steps, tools_used)
            elif "research_data" in enabled_tools:
                await self._execute_tool("research_data", question, context, steps, tools_used)

        else:
            logger.debug("WORKFLOW: route=default (search_documents or research_data)")
            if "search_documents" in enabled_tools:
                await self._execute_tool("search_documents", question, context, steps, tools_used)
            elif "research_data" in enabled_tools:
                await self._execute_tool("research_data", "general", context, steps, tools_used)

        if steps and "summarize_status" in enabled_tools and len(steps) < max_steps:
            logger.debug("WORKFLOW: running summarize_status on %d step results", len(steps))
            summary_input = " | ".join([step.get("result", "")[:100] for step in steps])
            await self._execute_tool("summarize_status", summary_input, context, steps, tools_used)

        logger.debug("WORKFLOW: complete | total_steps=%d | tools_used=%s", len(steps), tools_used)
        return StepFormatter.format_final_answer(steps)

    async def _execute_tool(
        self,
        tool_name: str,
        input_data: str,
        context: Dict[str, Any],
        steps: List[Dict[str, Any]],
        tools_used: List[str]
    ) -> str:
        """Execute a tool and record the step."""
        step_num = len(steps) + 1
        start = time.time()
        logger.debug("TOOL_EXEC: step=%d | tool=%s | input=%r", step_num, tool_name, str(input_data)[:80])

        try:
            tool = self.tools.get(tool_name)
            if not tool:
                result = f"Tool {tool_name} not found"
                logger.warning("TOOL_EXEC: tool not found | tool=%s", tool_name)
            else:
                result = await tool.execute(input_data, context)
                tools_used.append(tool_name)
                duration_ms = int((time.time() - start) * 1000)
                logger.info(
                    "TOOL_EXEC: success | step=%d | tool=%s | result_len=%d | duration_ms=%d",
                    step_num, tool_name, len(str(result)), duration_ms
                )
                logger.debug("TOOL_EXEC: result_preview=%r", str(result)[:120])

            step = StepFormatter.create_step_record(step_num, tool_name, input_data, result, time.time())
            steps.append(step)
            return result

        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {str(e)}"
            logger.error("TOOL_EXEC: error | step=%d | tool=%s | error=%s", step_num, tool_name, e, exc_info=True)
            step = StepFormatter.create_step_record(step_num, tool_name, input_data, error_msg, time.time(), error=True)
            steps.append(step)
            return error_msg
