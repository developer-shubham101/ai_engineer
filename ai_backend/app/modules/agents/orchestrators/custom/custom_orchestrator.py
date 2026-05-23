"""Agent orchestrator with LangChain integration."""

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
        self.max_steps = max_steps  # Hard limit from MOTIVATION.md
    
    def register_tool(self, tool: ITool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process agent request with safety constraints."""
        start_time = time.time()
        steps = []
        tools_used = []
        
        try:
            # Validate and limit steps
            max_steps = min(request.max_steps, self.max_steps)
            
            # Filter requested tools to available ones
            available_tools = self.get_available_tools()
            enabled_tools = [t for t in request.tools if t in available_tools] if request.tools else available_tools
            
            if not enabled_tools:
                return AgentResponse(
                    answer="No valid tools available for this request.",
                    steps=[],
                    tools_used=[],
                    debug_info={"error": "No valid tools"}
                )
            
            # Simple agent simulation (without full LangChain for now)
            answer = await self._simulate_agent_workflow(
                question=request.question,
                enabled_tools=enabled_tools,
                max_steps=max_steps,
                user=user,
                steps=steps,
                tools_used=tools_used
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            debug_info = None
            if request.debug:
                debug_info = {
                    "processing_time_ms": processing_time,
                    "available_tools": available_tools,
                    "enabled_tools": enabled_tools,
                    "max_steps": max_steps,
                    "actual_steps": len(steps)
                }
            
            return AgentResponse(
                answer=answer,
                steps=steps,
                tools_used=list(set(tools_used)),
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
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
        
        # Step 1: Analyze question and select initial tool
        if "ticket" in question.lower() and "get_user_tickets" in enabled_tools:
            result = await self._execute_tool("get_user_tickets", "current", context, steps, tools_used)
            
            # Step 2: If tickets found, get details
            if "TKT-" in result and "get_ticket_comments" in enabled_tools and len(steps) < max_steps:
                # Extract ticket ID (simple parsing)
                ticket_id = "TKT-001"  # Simplified for demo
                await self._execute_tool("get_ticket_comments", ticket_id, context, steps, tools_used)
        
        elif "search" in question.lower() or "document" in question.lower():
            if "search_documents" in enabled_tools:
                await self._execute_tool("search_documents", question, context, steps, tools_used)
        
        elif "web" in question.lower() or "internet" in question.lower() or "latest" in question.lower() or "current" in question.lower() or "news" in question.lower():
            if "web_search" in enabled_tools:
                result = await self._execute_tool("web_search", question, context, steps, tools_used)
                # Auto-scrape first URL if content is needed
                if "scrape_url" in enabled_tools and len(steps) < max_steps:
                    import re
                    urls = re.findall(r'https?://[^\s]+', result)
                    if urls:
                        await self._execute_tool("scrape_url", urls[0], context, steps, tools_used)
        
        elif "analyze" in question.lower() or "data" in question.lower():
            if "analyze_data" in enabled_tools:
                await self._execute_tool("analyze_data", question, context, steps, tools_used)
            elif "research_data" in enabled_tools:
                await self._execute_tool("research_data", question, context, steps, tools_used)
        
        else:
            # Default: try search first, then research
            if "search_documents" in enabled_tools:
                await self._execute_tool("search_documents", question, context, steps, tools_used)
            elif "research_data" in enabled_tools:
                await self._execute_tool("research_data", "general", context, steps, tools_used)
        
        # Final step: Summarize if we have results
        if steps and "summarize_status" in enabled_tools and len(steps) < max_steps:
            summary_input = " | ".join([step.get("result", "")[:100] for step in steps])
            await self._execute_tool("summarize_status", summary_input, context, steps, tools_used)
        
        # Compile final answer using utility
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
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                result = f"Tool {tool_name} not found"
            else:
                result = await tool.execute(input_data, context)
                tools_used.append(tool_name)
            
            step = StepFormatter.create_step_record(
                len(steps) + 1, tool_name, input_data, result, time.time()
            )
            steps.append(step)
            
            logger.debug(f"Executed tool {tool_name}: {result[:100]}...")
            return result
            
        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {str(e)}"
            step = StepFormatter.create_step_record(
                len(steps) + 1, tool_name, input_data, error_msg, time.time(), error=True
            )
            steps.append(step)
            logger.error(error_msg)
            return error_msg
