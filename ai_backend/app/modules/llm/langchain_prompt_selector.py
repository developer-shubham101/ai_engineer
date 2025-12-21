"""LangChain-based dynamic prompt template selector."""

import logging
import re
from dataclasses import dataclass
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# Core Abstractions and Base Classes

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Structured response model."""
    answer: str
    sources: List[str]
    confidence: str


@dataclass
class PromptData:
    """Data for a prompt."""
    user_question: str
    source_docs: str
    user_role: str = "user"
    department: str = "general"
    context_size: int = 0


class ConditionalPromptSelector:
    """Selects prompt template based on LLM context and user profile."""

    def __init__(self, template_manager=None):
        self.template_manager = template_manager
        # self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self, prompt_template_name: str) -> PromptTemplate:
        """Load the single prompt template."""
        if self.template_manager:
            logger.debug("Loading template from template manager")
            tmp_prompt_template_name = prompt_template_name or "personalized_chat"
            logger.debug("Using template name: %s", tmp_prompt_template_name)
            template_data = self.template_manager.get_template(tmp_prompt_template_name)

            if template_data:
                template_str = template_data["content"]

                # Use prompt_variables field if available, fallback to regex
                if 'prompt_variables' in template_data and template_data['prompt_variables']:
                    input_variables = [var.strip() for var in template_data['prompt_variables'].split('|') if var.strip()]
                else:
                    template_vars = re.findall(r"\{(\w+)}", template_str)
                    input_variables = [var for var in template_vars if var != "examples"]
                logger.debug("Extracted input variables: %s", input_variables)
                logger.debug("Template string: %s", template_str)

                return PromptTemplate(
                    input_variables=input_variables,
                    template=template_str
                    # partial_variables={"examples": ""}
                )

        # Fallback to loading from file if no manager or template not found
        logger.warning("Using fallback file template loading strategy")
        template_path = "app/modules/llm/prompt_templates/personalized_chat.txt"  # fallback path
        try:
            with open(template_path, "r") as f:
                template_str = f.read()

            template_vars = re.findall(r"\{(\w+)}", template_str)
            input_variables = [var for var in template_vars if var != "examples"]

            return PromptTemplate(
                input_variables=input_variables,
                template=template_str,
                partial_variables={"examples": ""}
            )
        except IOError as e:
            logger.error(f"Error loading template from {template_path}: {e}")
            return self.get_fallback_template()

    def format_prompt(self, prompt_data: dict, prompt_template_name: str) -> str:
        """Format the prompt with variables from a dataclass."""

        logger.debug("Formatting prompt with data: %s", prompt_data)
        logger.debug("Using template name: %s", prompt_template_name)
        prompt_template = self._load_prompt_template(prompt_template_name)
        logger.debug("Loaded prompt template: %s", prompt_template.template)

        # Filter for vars present in the template
        final_vars = {k: v for k, v in prompt_data.items() if k in prompt_template.input_variables}

        try:
            return prompt_template.format(**final_vars)
        except KeyError as e:
            logger.error("Missing template variable: %s", e)
            return f"Question: {prompt_data['user_question']}\nSources: {prompt_data['source_docs']}"

    def get_fallback_template(self) -> PromptTemplate:
        """Get ironclad JSON template for retry."""
        return PromptTemplate(
            input_variables=["user_question", "source_docs"],
            template="""Answer this question using the sources provided. You MUST respond with valid JSON in this exact format:

{{"answer": "your answer here", "sources": ["source1", "source2"], "confidence": "low"}}

Sources: {source_docs}
Question: {user_question}

JSON Response:"""
        )
