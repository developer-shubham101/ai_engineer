"""Debate workflow: Advocate vs Critic, moderated by Moderator."""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ....interfaces import AgentResponse
from ...utils import run_team

logger = logging.getLogger(__name__)


async def execute_debate_workflow(
    model_client: Any,
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """Three-agent debate: Advocate vs Critic, moderated by Moderator."""
    logger.debug("[debate] START query_len=%d tools=%s max_steps=%d",
                 len(query), [t.__name__ for t in tools], max_steps)
    advocate = AssistantAgent(
        name="Advocate",
        system_message="You argue FOR the given topic with strong supporting evidence. Be concise.",
        model_client=model_client,
        tools=tools or None,
    )
    critic = AssistantAgent(
        name="Critic",
        system_message="You argue AGAINST the given topic with counterarguments. Be concise.",
        model_client=model_client,
        tools=tools or None,
    )
    moderator = AssistantAgent(
        name="Moderator",
        system_message="Moderate the debate and provide a final balanced summary. Be concise.",
        model_client=model_client,
    )

    team = RoundRobinGroupChat(
        participants=[advocate, critic, moderator],
        termination_condition=MaxMessageTermination(max_messages=max_steps),
    )
    final_result, steps, tools_used = await run_team(team, f"Debate topic: {query}")
    logger.debug("[debate] DONE steps=%d tools_used=%s answer_len=%d",
                 len(steps), tools_used, len(final_result))

    return AgentResponse(
        answer=final_result,
        steps=steps,
        tools_used=list(tools_used) or [t.__name__ for t in tools],
        final_step=True,
    )
