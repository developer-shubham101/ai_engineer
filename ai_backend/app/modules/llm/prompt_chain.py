"""Chain of Responsibility pattern for prompt building."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Context passed through the prompt chain."""
    user: Optional[Dict[str, Any]] = None
    question: str = ""
    context: str = ""
    category: Optional[str] = None
    session_id: Optional[str] = None
    enhanced_query: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    final_prompt: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.enhanced_query:
            self.enhanced_query = self.question


class PromptHandler(ABC):
    """Abstract prompt handler."""
    
    def __init__(self):
        self._next_handler: Optional[PromptHandler] = None
    
    def set_next(self, handler: 'PromptHandler') -> 'PromptHandler':
        """Set next handler in chain."""
        self._next_handler = handler
        return handler
    
    async def handle(self, context: PromptContext) -> PromptContext:
        """Handle prompt building step."""
        context = await self.process(context)
        
        if self._next_handler:
            return await self._next_handler.handle(context)
        
        return context
    
    @abstractmethod
    async def process(self, context: PromptContext) -> PromptContext:
        """Process this step of prompt building."""
        pass


class SystemPromptHandler(PromptHandler):
    """Builds system prompt."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        role = context.user.get("role", "Guest") if context.user else "Guest"
        dept = context.user.get("department", "General") if context.user else "General"
        
        system_parts = [
            f"You are an AI assistant for {dept} department.",
            f"User role: {role}",
            "Provide accurate, helpful responses based on the context."
        ]
        
        if context.category:
            system_parts.append(f"Focus on {context.category} related information.")
        
        context.system_prompt = " ".join(system_parts)
        return context


class QueryEnhancementHandler(PromptHandler):
    """Enhances query with user context."""
    
    def __init__(self, session_manager=None):
        super().__init__()
        self.session_manager = session_manager
    
    async def process(self, context: PromptContext) -> PromptContext:
        enhanced = context.question
        
        # Add user profile
        if context.user:
            role = context.user.get("role", "")
            dept = context.user.get("department", "")
            if role or dept:
                enhanced += f" [User: {role} in {dept}]"
        
        # Add sentiment if available
        if self.session_manager and context.session_id:
            try:
                messages = self.session_manager.get_recent_messages(context.session_id, limit=3)
                if messages:
                    last_sentiment = messages[-1].get("sentiment")
                    if last_sentiment and last_sentiment != "neutral":
                        enhanced += f" [Mood: {last_sentiment}]"
            except:
                pass
        
        # Add category
        if context.category:
            enhanced += f" [Category: {context.category}]"
        
        context.enhanced_query = enhanced
        return context


class UserPromptHandler(PromptHandler):
    """Builds user prompt."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        context.user_prompt = f"Question: {context.enhanced_query}\n\nContext:\n{context.context}"
        return context


class PersonalizationHandler(PromptHandler):
    """Adds personalization to prompts."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        if context.user:
            name = context.user.get("username", "User")
            context.system_prompt += f" Address the user as {name}."
        return context


class SecurityHandler(PromptHandler):
    """Adds security instructions."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        role = context.user.get("role", "Guest") if context.user else "Guest"
        
        if role in ["Guest", "Employee"]:
            context.system_prompt += " Do not reveal sensitive company information."
        
        return context


class FinalPromptHandler(PromptHandler):
    """Combines all prompts into final prompt."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        context.final_prompt = f"{context.system_prompt}\n\n{context.user_prompt}"
        return context


class PromptChain:
    """Manages the prompt building chain."""
    
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.available_handlers = {
            'system': SystemPromptHandler(),
            'personalization': PersonalizationHandler(),
            'security': SecurityHandler(),
            'query_enhancement': QueryEnhancementHandler(self.session_manager),
            'user_prompt': UserPromptHandler(),
            'final': FinalPromptHandler()
        }
    
    def _build_dynamic_chain(self, context: PromptContext) -> PromptHandler:
        """Build chain dynamically based on available context."""
        handlers = []
        
        # Always start with system
        handlers.append(self.available_handlers['system'])
        
        # Add personalization if user exists
        if context.user:
            handlers.append(self.available_handlers['personalization'])
            handlers.append(self.available_handlers['security'])
        
        # Add query enhancement if we have user or session
        if context.user or context.session_id or context.category:
            handlers.append(self.available_handlers['query_enhancement'])
        
        # Always add user prompt and final
        handlers.append(self.available_handlers['user_prompt'])
        handlers.append(self.available_handlers['final'])
        
        # Chain them together
        for i in range(len(handlers) - 1):
            handlers[i].set_next(handlers[i + 1])
        
        return handlers[0]
    
    async def build_prompt(self, question: str, context: str = "", 
                          user: Optional[Dict[str, Any]] = None,
                          session_id: Optional[str] = None,
                          category: Optional[str] = None) -> str:
        """Build prompt with dynamic chain based on available data."""
        prompt_context = PromptContext(
            user=user,
            question=question,
            context=context,
            category=category,
            session_id=session_id
        )
        
        # Build dynamic chain based on context
        chain = self._build_dynamic_chain(prompt_context)
        result = await chain.handle(prompt_context)
        return result.final_prompt
    
    async def build_enhanced_query(self, question: str,
                                  user: Optional[Dict[str, Any]] = None,
                                  session_id: Optional[str] = None,
                                  category: Optional[str] = None) -> str:
        """Build enhanced query for document retrieval."""
        prompt_context = PromptContext(
            user=user,
            question=question,
            category=category,
            session_id=session_id
        )
        
        # Only run query enhancement handler
        query_handler = QueryEnhancementHandler(self.session_manager)
        result = await query_handler.handle(prompt_context)
        return result.enhanced_query
    
    def add_handler(self, name: str, handler: PromptHandler):
        """Add custom handler."""
        self.available_handlers[name] = handler
    
    def remove_handler(self, name: str):
        """Remove handler."""
        if name in self.available_handlers:
            del self.available_handlers[name]
    
    def add_handler(self, handler: PromptHandler, position: int = -1):
        """Add custom handler to chain."""
        # Implementation for dynamic chain modification
        pass
    
    def remove_handler(self, handler_type: type):
        """Remove handler from chain."""
        # Implementation for dynamic chain modification
        pass