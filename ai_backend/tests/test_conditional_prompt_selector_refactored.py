"""Tests for the refactored ConditionalPromptSelector class."""

import pytest
from app.modules.llm.langchain_prompt_selector import ConditionalPromptSelector, PromptData

@pytest.fixture
def selector():
    """Returns a ConditionalPromptSelector instance for testing."""
    return ConditionalPromptSelector(templates_dir="app/modules/llm/prompt_templates")

def test_initialization(selector):
    """Test that the selector initializes correctly."""
    assert selector.embedding_manager is not None
    assert selector.templates_dir == "app/modules/llm/prompt_templates"
    assert selector.prompt_template is not None

def test_load_prompt_template(selector):
    """Test that the prompt template is loaded correctly."""
    assert "SYSTEM: Enterprise assistant for Saarthi Infotech" in selector.prompt_template.template

def test_get_prompt_template(selector):
    """Test getting a prompt template."""
    assert selector.prompt_template is not None
    assert "user_profile_summary" not in selector.prompt_template.input_variables

def test_format_prompt(selector):
    """Test formatting a prompt."""
    prompt_data = PromptData(
        user_question="How do I deploy to production?",
        source_docs="deploy_guide_002",
        user_role="Developer",
        department="Engineering",
    )
    formatted_prompt = selector.format_prompt(prompt_data)
    
    assert "Role: Developer" in formatted_prompt
    assert "Department: Engineering" in formatted_prompt
    assert "USER QUESTION:\nHow do I deploy to production?" in formatted_prompt
    assert "deploy_guide_002" in formatted_prompt
