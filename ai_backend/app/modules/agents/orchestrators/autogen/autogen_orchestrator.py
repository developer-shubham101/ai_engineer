"""AutoGen-based agent orchestrator for multi-agent conversations using AutoGen v0.4."""

import logging
from typing import Dict, Any, Optional, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ...interfaces import IAgentOrchestrator, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class AutoGenOrchestrator(IAgentOrchestrator):
    """AutoGen-based multi-agent orchestrator using v0.4 API."""

    def __init__(self):
        # Initialize the model client pointing to local llama-server
        self.model_client = OpenAIChatCompletionClient(
            model="mistral-7b-instruct",  # informative name
            base_url="http://127.0.0.1:8080/v1",
            api_key="placeholder",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "structured_output": False,
                "family": "unknown",
            },
        )

    async def process_request(self, request: AgentRequest, user: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process agent request using AutoGen."""
        try:
            # Check if workflow type is mentioned in the question (simple heuristic for now)
            # or default to debate if acceptable. 
            # Ideally AgentRequest should have workflow_type. 
            # For this review fix, we'll default to debate or simple detection.
            query = request.question
            lower_q = query.lower()

            if "research" in lower_q:
                return await self._execute_research_workflow(query)
            else:
                return await self._execute_debate_workflow(query)

        except Exception as e:
            logger.error(f"AutoGen workflow failed: {e}")
            return AgentResponse(
                answer=f"Workflow failed: {str(e)}",
                steps=[],
                tools_used=[],
                final_step=True
            )

    def register_tool(self, tool: Any) -> None:
        """Register a tool (Not implemented for v0.4 team yet)."""
        pass

    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return []

    async def _execute_debate_workflow(self, query: str) -> AgentResponse:
        """Execute debate workflow with AutoGen agents."""

        # 1. Create Agents
        advocate = AssistantAgent(
            name="Advocate",
            system_message="You argue FOR the given topic with strong supporting evidence.",
            model_client=self.model_client,
        )

        critic = AssistantAgent(
            name="Critic",
            system_message="You argue AGAINST the given topic with counterarguments.",
            model_client=self.model_client,
        )

        moderator = AssistantAgent(
            name="Moderator",
            system_message="You moderate the debate and provide final balanced summary.",
            model_client=self.model_client,
        )

        # 2. Define Termination Condition
        # Stop after 6 messages or if "TERMINATE" is mentioned (optional)
        termination = MaxMessageTermination(max_messages=6)

        # 3. Create Team
        team = RoundRobinGroupChat(
            participants=[advocate, critic, moderator],
            termination_condition=termination
        )

        # 4. Run Team
        stream = team.run_stream(task=f"Let's debate: {query}")

        steps = []
        final_result = ""

        # Collect output
        async for message in stream:
            # We can log steps here
            if hasattr(message, 'content'):
                # This might need adjustment based on exact v0.4 message structure
                steps.append(f"{message.source}: {str(message.content)[:100]}...")

            # Keep track of the last response as result
            if hasattr(message, 'content'):
                final_result = str(message.content)

        return AgentResponse(
            answer=final_result,
            steps=[{"source": "autogen", "content": s} for s in steps],
            tools_used=[],
            final_step=True
        )

    async def _execute_research_workflow(self, query: str) -> AgentResponse:
        """Execute research workflow with AutoGen agents."""

        researcher = AssistantAgent(
            name="Researcher",
            system_message="You research and gather information on the given topic.",
            model_client=self.model_client,
        )

        analyst = AssistantAgent(
            name="Analyst",
            system_message="You analyze the research and identify key insights.",
            model_client=self.model_client,
        )

        # Stop after 4 loops
        termination = MaxMessageTermination(max_messages=4)

        team = RoundRobinGroupChat(
            participants=[researcher, analyst],
            termination_condition=termination
        )

        stream = team.run_stream(task=f"Research this topic: {query}")

        steps = []
        final_result = ""

        async for message in stream:
            if hasattr(message, 'content'):
                steps.append(f"{message.source}: {str(message.content)[:100]}...")
                final_result = str(message.content)

        return AgentResponse(
            answer=final_result,
            steps=[{"source": "autogen", "content": s} for s in steps],
            tools_used=[],
            final_step=True
        )
