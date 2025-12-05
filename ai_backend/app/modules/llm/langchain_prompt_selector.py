"""LangChain-based dynamic prompt template selector."""

from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    """Structured response model."""
    answer: str
    sources: List[str]
    confidence: str

class ConditionalPromptSelector:
    """Selects prompt template based on LLM context and user profile."""

    def __init__(self, templates_dir: str = "app/modules/llm/prompt_templates"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.templates_dir = templates_dir
        self.examples = self._load_examples()
        self.example_selector = self._create_example_selector()

    def _load_examples(self) -> dict[str, list[dict[str, str]]]:
        """Load few-shot examples from a JSON file."""
        examples_path = os.path.join(self.templates_dir, "prompt_examples.json")
        try:
            with open(examples_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading examples from {examples_path}: {e}")
            return {}

    def _create_example_selector(self) -> Optional[SemanticSimilarityExampleSelector]:
        """Create semantic example selector."""
        if not self.examples:
            return None
            
        all_examples = [ex for dept_examples in self.examples.values() for ex in dept_examples]

        if not all_examples:
            return None

        return SemanticSimilarityExampleSelector.from_examples(
            all_examples,
            self.embeddings,
            FAISS,
            k=2
        )

    def _load_base_template(self, template_name: str) -> str:
        """Load a base template from a file."""
        template_path = os.path.join(self.templates_dir, template_name)
        try:
            with open(template_path, "r") as f:
                return f.read()
        except IOError as e:
            logger.error(f"Error loading template from {template_path}: {e}")
            return ""

    def get_prompt_template(self, prompt_data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Get appropriate prompt template based on dynamic data."""
        context_size = prompt_data.get("context_size", 0)
        department = prompt_data.get("department", "")
        user_question = prompt_data.get("user_question", "")

        # 1. Select template file
        template_name = "long_context_template.txt" if context_size >= 4000 else "short_context_template.txt"
        base_template = self._load_base_template(template_name)
        if not base_template:
            return None

        # 2. Get relevant examples
        examples_text = ""
        if self.example_selector:
            dept_examples = self.examples.get(department, [])
            if dept_examples:
                selected_examples = self.example_selector.select_examples({"q": user_question})
                dept_selected = [ex for ex in selected_examples if ex in dept_examples]
                examples_to_use = dept_selected[:2] if dept_selected else selected_examples[:2]
            else:
                examples_to_use = self.example_selector.select_examples({"q": user_question})[:2]
            
            if examples_to_use:
                examples_text = "EXAMPLES:\n" + "\n".join([
                    f"Q: {ex['q']}\nA: {ex['a']}" for ex in examples_to_use
                ]) + "\n"

        # 3. Dynamically determine input variables from template
        template_vars = re.findall(r"\{(\w+)\}", base_template)
        input_variables = [var for var in template_vars if var != "examples"]

        return PromptTemplate(
            input_variables=input_variables,
            template=base_template,
            partial_variables={"examples": examples_text}
        )

    def format_prompt(self, template: PromptTemplate, prompt_data: Dict[str, Any]) -> str:
        """Format the prompt with variables from a dictionary."""
        format_vars = {k: v for k, v in prompt_data.items() if k in template.input_variables}
        
        try:
            return template.format(**format_vars)
        except KeyError as e:
            logger.error("Missing template variable: %s", e)
            # Fallback to a simple format
            return f"Question: {prompt_data.get('user_question', '')}\nSources: {prompt_data.get('source_docs', '')}"


    def validate_response(self, response_text: str) -> Optional[RAGResponse]:
        """Validate and parse JSON response."""
        try:
            # Try to extract JSON from response
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                
                data = json.loads(json_str)
                return RAGResponse(**data)
        except Exception:
            pass
        
        return None

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


# Usage example
def create_dynamic_prompt(prompt_data: Dict[str, Any]) -> str:
    """Create dynamic prompt based on context and user profile."""
    selector = ConditionalPromptSelector()
    
    template = selector.get_prompt_template(prompt_data)
    
    if not template:
        return "Error: Could not generate a prompt template."

    return selector.format_prompt(
        template=template,
        prompt_data=prompt_data
    )