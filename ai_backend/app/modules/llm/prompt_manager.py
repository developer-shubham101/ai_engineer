"""Prompt management implementation."""

from typing import Dict, Any, List, Optional
import logging

from .interfaces import IPromptManager
from ..config.constants import MAX_CONTEXT_TOKENS, MAX_SYSTEM_TOKENS

logger = logging.getLogger(__name__)


class PromptManager(IPromptManager):
    """Prompt management implementation."""
    
    def __init__(self):
        self.system_template = """You are a helpful AI assistant for {company_name}. 
Role: {role} | Dept: {department}
{tone_guidance}
Answer based on provided context. Be concise and accurate."""
    
    async def build_system_prompt(self, user: Dict[str, Any], context: str, category: Optional[str] = None) -> str:
        """Build system prompt."""
        role = user.get("role", "Employee")
        department = user.get("department", "General")
        company_name = "Your Company"  # Could be configurable
        
        # Build tone guidance based on user profile or session history
        tone_guidance = self._build_tone_guidance(user)
        
        system_prompt = self.system_template.format(
            company_name=company_name,
            role=role,
            department=department,
            tone_guidance=tone_guidance
        )
        
        # Add category-specific instructions
        if category:
            system_prompt += f"\nFocus on: {category}"
        
        # Truncate if too long
        if self.estimate_tokens(system_prompt) > MAX_SYSTEM_TOKENS:
            system_prompt = self.truncate_context(system_prompt, MAX_SYSTEM_TOKENS)
        
        return system_prompt
    
    async def build_user_prompt(self, question: str, context: str) -> str:
        """Build user prompt."""
        if context.strip():
            return f"Context:\n{context}\n\nQuestion: {question}"
        else:
            return f"Question: {question}"
    
    async def build_full_prompt(self, system_prompt: str, user_prompt: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Build complete prompt."""
        prompt_parts = [system_prompt]
        
        # Add conversation history if provided
        if conversation_history:
            history_text = self._format_conversation_history(conversation_history)
            if history_text:
                prompt_parts.append(f"Recent conversation:\n{history_text}")
        
        prompt_parts.append(user_prompt)
        
        return "\n\n".join(prompt_parts)
    
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit token limit."""
        estimated_tokens = self.estimate_tokens(context)
        
        if estimated_tokens <= max_tokens:
            return context
        
        # Simple truncation - take first part of context
        # In production, you might want smarter truncation
        words = context.split()
        target_words = int(len(words) * (max_tokens / estimated_tokens))
        
        truncated = " ".join(words[:target_words])
        
        # Add truncation indicator
        if len(truncated) < len(context):
            truncated += "\n[... content truncated ...]"
        
        return truncated
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _build_tone_guidance(self, user: Dict[str, Any]) -> str:
        """Build tone guidance based on user."""
        role = user.get("role", "Employee")
        
        tone_map = {
            "SuperAdmin": "Be direct and comprehensive.",
            "Manager": "Be professional and detailed.",
            "HR": "Be empathetic and policy-focused.",
            "Employee": "Be helpful and clear.",
            "Guest": "Be welcoming and informative."
        }
        
        return tone_map.get(role, "Be helpful and professional.")
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return ""
        
        formatted_messages = []
        for msg in history[-3:]:  # Only include last 3 messages
            speaker = msg.get("speaker", "unknown")
            content = msg.get("content", "")
            
            if speaker == "user":
                formatted_messages.append(f"User: {content}")
            elif speaker == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        
        return "\n".join(formatted_messages)


class OptimizedPromptManager(PromptManager):
    """Optimized prompt manager with advanced features."""
    
    def __init__(self):
        super().__init__()
        # Ultra-compact system template
        self.system_template = "AI assistant for {company}. Role: {role}. {tone}. Use context provided."
    
    async def build_system_prompt(self, user: Dict[str, Any], context: str, category: Optional[str] = None) -> str:
        """Build optimized system prompt."""
        role = user.get("role", "Employee")
        company = "Company"  # Shortened
        
        # Ultra-compact tone guidance
        tone = self._get_compact_tone(role)
        
        system_prompt = self.system_template.format(
            company=company,
            role=role,
            tone=tone
        )
        
        if category:
            system_prompt += f" Focus: {category}."
        
        return system_prompt
    
    def _get_compact_tone(self, role: str) -> str:
        """Get compact tone guidance."""
        tone_map = {
            "SuperAdmin": "Direct",
            "Manager": "Professional", 
            "HR": "Empathetic",
            "Employee": "Helpful",
            "Guest": "Welcoming"
        }
        return tone_map.get(role, "Professional")
    
    def estimate_tokens(self, text: str) -> int:
        """More accurate token estimation."""
        # Better estimation considering punctuation and spaces
        words = text.split()
        chars = len(text)
        
        # Rough approximation: average of word-based and char-based estimates
        word_estimate = len(words) * 1.3  # Average 1.3 tokens per word
        char_estimate = chars / 3.5  # Average 3.5 characters per token
        
        return int((word_estimate + char_estimate) / 2)