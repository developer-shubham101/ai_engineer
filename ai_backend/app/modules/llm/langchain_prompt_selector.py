"""LangChain-based dynamic prompt template selector."""

from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
import json
import logging
logger = logging.getLogger(__name__)
class RAGResponse(BaseModel):
    """Structured response model."""
    answer: str
    sources: List[str]
    confidence: str


class ConditionalPromptSelector:
    """Selects prompt template based on LLM context and user profile."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.examples = self._load_examples()
        self.example_selector = self._create_example_selector()
    
    @staticmethod
    def _load_examples() -> dict[str, list[dict[str, str]]]:
        """Load few-shot examples by department."""
        return {
            "HR": [
                {"q": "What is the leave policy?", "a": '{"answer": "Annual leave is 20 days per year", "sources": ["hr_policy_001"], "confidence": "high"}'},
                {"q": "How do I apply for sick leave?", "a": '{"answer": "Submit form HR-101 to your manager", "sources": ["hr_forms_002"], "confidence": "high"}'}
            ],
            "Engineering": [
                {"q": "What are our coding standards?", "a": '{"answer": "Follow PEP 8 for Python, ESLint for JS", "sources": ["dev_guide_001"], "confidence": "high"}'},
                {"q": "How do I deploy to production?", "a": '{"answer": "Use CI/CD pipeline via Jenkins", "sources": ["deploy_guide_002"], "confidence": "medium"}'}
            ],
            "Finance": [
                {"q": "What is the expense limit?", "a": '{"answer": "Up to $500 without approval", "sources": ["finance_policy_001"], "confidence": "high"}'},
                {"q": "How do I submit receipts?", "a": '{"answer": "Upload to expense portal within 30 days", "sources": ["expense_guide_002"], "confidence": "high"}'}
            ]
        }
    
    def _create_example_selector(self) -> SemanticSimilarityExampleSelector:
        """Create semantic example selector."""
        all_examples = []
        for dept_examples in self.examples.values():
            all_examples.extend(dept_examples)
        
        return SemanticSimilarityExampleSelector.from_examples(
            all_examples,
            self.embeddings,
            FAISS,
            k=2
        )
    
    def _get_base_template(self, context_size: int) -> str:
        """Get base template based on context size."""
        if context_size >= 4000:
            return """SYSTEM: You are a helpful enterprise assistant for Saarthi Infotech. Only use documents in SOURCE_DOCS. Never invent facts. If answer not supported, say "I don't know" and list sources checked.

CONTEXT:
- Role: {user_role}
- Department: {department}
- Profile: {user_profile_summary}

{examples}

SOURCE_DOCS:
{source_docs}

INSTRUCTION:
Answer using only SOURCE_DOCS. Keep answer <= {max_tokens} tokens.
Return JSON: {{"answer":"...", "sources":["doc_id1","doc_id2"], "confidence": "low|medium|high"}}

USER QUESTION:
{user_question}"""
        else:
            return """SYSTEM: Enterprise assistant for Saarthi Infotech. Use only SOURCE_DOCS.

Role: {user_role} | Dept: {department}

{examples}

SOURCES: {source_docs}

Return JSON: {{"answer":"...", "sources":[], "confidence":"low|medium|high"}}

Q: {user_question}"""
    
    def get_prompt_template(self, context_size: int, user_role: str, department: str, 
                           user_question: str, max_tokens: int = 256) -> PromptTemplate:
        """Get appropriate prompt template."""
        
        # Get relevant examples
        dept_examples = self.examples.get(department, [])
        if dept_examples:
            selected_examples = self.example_selector.select_examples({"q": user_question})
            # Filter to department examples if available
            dept_selected = [ex for ex in selected_examples if ex in dept_examples]
            examples_to_use = dept_selected[:2] if dept_selected else selected_examples[:2]
        else:
            examples_to_use = self.example_selector.select_examples({"q": user_question})[:2]
        
        # Format examples
        examples_text = ""
        if examples_to_use:
            examples_text = "EXAMPLES:\n" + "\n".join([
                f"Q: {ex['q']}\nA: {ex['a']}" for ex in examples_to_use
            ]) + "\n"
        
        base_template = self._get_base_template(context_size)
        
        # Check which variables are actually in the template
        template_vars = ["user_role", "department", "source_docs", "user_question"]
        if "{user_profile_summary}" in base_template:
            template_vars.append("user_profile_summary")
        if "{max_tokens}" in base_template:
            template_vars.append("max_tokens")
        
        return PromptTemplate(
            input_variables=template_vars,
            template=base_template,
            partial_variables={"examples": examples_text}
        )
    
    def format_prompt(self, template: PromptTemplate, user_role: str, department: str,
                     user_profile_summary: str, source_docs: str, user_question: str,
                     max_tokens: int = 256) -> str:
        """Format the prompt with variables."""
        # Only pass variables that are in the template's input_variables
        format_vars = {
            "user_role": user_role,
            "department": department,
            "source_docs": source_docs,
            "user_question": user_question
        }
        
        # Add optional variables only if they're in the template
        if "user_profile_summary" in template.input_variables:
            format_vars["user_profile_summary"] = str(user_profile_summary)
        if "max_tokens" in template.input_variables:
            format_vars["max_tokens"] = str(max_tokens)
        
        try:
            return template.format(**format_vars)
        except KeyError as e:
            logger.error("Missing template variable: %s", e)
            return f"Role: {user_role}\nDept: {department}\nSources: {source_docs}\nQ: {user_question}\nReturn JSON with answer, sources, confidence."
    
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
def create_dynamic_prompt(context_size: int, user_role: str, department: str,
                         user_question: str, source_docs: str, 
                         user_profile_summary: str = "", max_tokens: int = 256) -> str:
    """Create dynamic prompt based on context and user profile."""
    selector = ConditionalPromptSelector()
    
    template = selector.get_prompt_template(
        context_size=context_size,
        user_role=user_role,
        department=department,
        user_question=user_question,
        max_tokens=max_tokens
    )
    
    return selector.format_prompt(
        template=template,
        user_role=user_role,
        department=department,
        user_profile_summary=user_profile_summary,
        source_docs=source_docs,
        user_question=user_question,
        max_tokens=max_tokens
    )