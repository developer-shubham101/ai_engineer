"""Prompt evaluation workflow: PromptParser → CriteriaJudge → Improver → EvalReporter."""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from ....interfaces import AgentResponse
from ...utils import run_team

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation criteria used in agent system messages and the reporter template
# ---------------------------------------------------------------------------
_CRITERIA = [
    "Clarity        — Is the prompt unambiguous and easy to understand?",
    "Specificity    — Does it provide enough detail to constrain the output?",
    "Context        — Does it supply necessary background or role framing?",
    "Safety         — Does it avoid harmful, biased, or policy-violating instructions?",
    "Token Efficiency — Is it concise without sacrificing meaning?",
]

_CRITERIA_BLOCK = "\n".join(f"  {c}" for c in _CRITERIA)

_SCORE_FORMAT = (
    "Return a JSON object with this exact shape — no prose, no markdown fences:\n"
    '{"scores": {"clarity": 0-10, "specificity": 0-10, "context": 0-10, '
    '"safety": 0-10, "token_efficiency": 0-10}, '
    '"issues": ["issue1", "issue2", ...], '
    '"overall": 0-10}'
)


async def execute_prompt_evaluation_workflow(
    model_client: Any,
    query: str,
    tools: List[Callable],
    max_steps: int,
) -> AgentResponse:
    """4-agent prompt evaluation pipeline.

    Agents:
      1. PromptParser   — extracts intent, variables, constraints, role framing
      2. CriteriaJudge  — scores the prompt on 5 criteria and lists issues
      3. Improver       — rewrites the prompt addressing every identified issue
      4. EvalReporter   — assembles the final structured evaluation report
    """
    logger.debug("[prompt_eval] START query_len=%d max_steps=%d", len(query), max_steps)

    prompt_parser = AssistantAgent(
        name="PromptParser",
        system_message=(
            "You are a prompt analysis expert.\n\n"
            "Given a prompt, extract and clearly state:\n"
            "- Intent: what the prompt is trying to achieve\n"
            "- Target model: who/what will receive this prompt (LLM, agent, API, etc.)\n"
            "- Variables: any placeholders or dynamic parts (e.g. {topic}, {user_name})\n"
            "- Constraints: explicit rules or limits stated in the prompt\n"
            "- Role framing: any persona or role assigned to the model\n"
            "- Missing context: what background information is absent\n\n"
            "Be concise. Use bullet points."
        ),
        model_client=model_client,
    )

    criteria_judge = AssistantAgent(
        name="CriteriaJudge",
        system_message=(
            "You are a strict prompt quality evaluator.\n\n"
            "Score the prompt on these 5 criteria (0 = terrible, 10 = perfect):\n"
            f"{_CRITERIA_BLOCK}\n\n"
            "Also list every concrete issue you find (be specific, not generic).\n\n"
            f"{_SCORE_FORMAT}"
        ),
        model_client=model_client,
    )

    improver = AssistantAgent(
        name="Improver",
        system_message=(
            "You are a prompt engineering expert.\n\n"
            "You will receive:\n"
            "- The original prompt\n"
            "- A parsed breakdown of its structure\n"
            "- A list of issues identified by the judge\n\n"
            "Your job:\n"
            "1. Rewrite the prompt to fix every listed issue\n"
            "2. Preserve the original intent exactly\n"
            "3. Keep the improved prompt concise — do not pad it\n"
            "4. After the improved prompt, add a short 'Changes made:' bullet list\n\n"
            "Format:\n"
            "IMPROVED PROMPT:\n"
            "<the rewritten prompt>\n\n"
            "CHANGES MADE:\n"
            "- <change 1>\n"
            "- <change 2>\n"
            "..."
        ),
        model_client=model_client,
    )

    eval_reporter = AssistantAgent(
        name="EvalReporter",
        system_message=(
            "You are a technical report writer.\n\n"
            "Assemble a final structured prompt evaluation report using ALL prior agent outputs.\n\n"
            "Report structure (use these exact headings):\n"
            "## Prompt Evaluation Report\n\n"
            "### Original Prompt\n"
            "<quote the original prompt>\n\n"
            "### Parsed Structure\n"
            "<summary from PromptParser>\n\n"
            "### Scores\n"
            "| Criterion | Score | Notes |\n"
            "|---|---|---|\n"
            "| Clarity | X/10 | ... |\n"
            "| Specificity | X/10 | ... |\n"
            "| Context | X/10 | ... |\n"
            "| Safety | X/10 | ... |\n"
            "| Token Efficiency | X/10 | ... |\n"
            "| **Overall** | **X/10** | |\n\n"
            "### Issues Found\n"
            "<numbered list of issues from CriteriaJudge>\n\n"
            "### Improved Prompt\n"
            "<improved prompt from Improver>\n\n"
            "### Changes Made\n"
            "<bullet list of changes from Improver>\n\n"
            "### Verdict\n"
            "One paragraph: is the original prompt usable as-is, needs minor fixes, "
            "or needs a full rewrite? Justify based on the overall score."
        ),
        model_client=model_client,
    )

    team = RoundRobinGroupChat(
        participants=[prompt_parser, criteria_judge, improver, eval_reporter],
        termination_condition=MaxMessageTermination(max_messages=max_steps),
    )

    task = (
        f"Evaluate the following prompt:\n\n"
        f"---\n{query}\n---\n\n"
        "PromptParser: parse the structure.\n"
        "CriteriaJudge: score it and list issues.\n"
        "Improver: rewrite it fixing all issues.\n"
        "EvalReporter: produce the final evaluation report."
    )

    final_result, steps, tools_used = await run_team(team, task)
    logger.debug("[prompt_eval] DONE steps=%d answer_len=%d", len(steps), len(final_result))

    return AgentResponse(
        answer=final_result,
        steps=steps,
        tools_used=list(tools_used),
        final_step=True,
        debug_info={"workflow": "prompt_evaluation", "agents": ["PromptParser", "CriteriaJudge", "Improver", "EvalReporter"]},
    )
