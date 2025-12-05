"""Tests for the ConditionalPromptSelector class."""

import pytest
from app.modules.llm.langchain_prompt_selector import ConditionalPromptSelector

@pytest.fixture
def selector():
    """Returns a ConditionalPromptSelector instance for testing."""
    return ConditionalPromptSelector(templates_dir="app/modules/llm/prompt_templates")

def test_initialization(selector):
    """Test that the selector initializes correctly."""
    assert selector.embeddings is not None
    assert selector.templates_dir == "app/modules/llm/prompt_templates"
    assert selector.examples is not None
    assert selector.example_selector is not None

def test_load_examples(selector):
    """Test that examples are loaded correctly."""
    assert "HR" in selector.examples
    assert "Engineering" in selector.examples
    assert "Finance" in selector.examples
    assert len(selector.examples["HR"]) > 0

def test_load_base_template(selector):
    """Test that base templates are loaded correctly."""
    long_template = selector._load_base_template("long_context_template.txt")
    short_template = selector._load_base_template("short_context_template.txt")
    assert "SYSTEM: You are a helpful enterprise assistant" in long_template
    assert "SYSTEM: Enterprise assistant for Saarthi Infotech" in short_template

def test_get_prompt_template_long_context(selector):
    """Test getting a prompt template for a long context."""
    prompt_data = {
        "context_size": 5000,
        "department": "Engineering",
        "user_question": "What are our coding standards?"
    }
    template = selector.get_prompt_template(prompt_data)
    assert template is not None
    assert "user_profile_summary" in template.input_variables

def test_get_prompt_template_short_context(selector):
    """Test getting a prompt template for a short context."""
    prompt_data = {
        "context_size": 1000,
        "department": "HR",
        "user_question": "What is the leave policy?"
    }
    template = selector.get_prompt_template(prompt_data)
    assert template is not None
    assert "user_profile_summary" not in template.input_variables

def test_format_prompt(selector):
    """Test formatting a prompt."""
    prompt_data = {
        "context_size": 5000,
        "user_role": "Developer",
        "department": "Engineering",
        "user_question": "How do I deploy to production?",
        "source_docs": "deploy_guide_002",
        "user_profile_summary": "A developer profile",
        "max_tokens": "100"
    }
    template = selector.get_prompt_template(prompt_data)
    formatted_prompt = selector.format_prompt(template, prompt_data)
    
    assert "Role: Developer" in formatted_prompt
    assert "Department: Engineering" in formatted_prompt
    assert "USER QUESTION:\nHow do I deploy to production?" in formatted_prompt
    assert "deploy_guide_002" in formatted_prompt
    assert "A developer profile" in formatted_prompt
    assert "100" in formatted_prompt
