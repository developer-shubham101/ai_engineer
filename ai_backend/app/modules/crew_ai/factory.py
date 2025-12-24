"""Factory for CrewAI components."""

from typing import Optional
from .interfaces import ICrewOrchestrator
from .orchestrator import CrewOrchestrator


class CrewOrchestratorFactory:
    """Factory for creating CrewAI orchestrators."""
    
    @staticmethod
    def create_orchestrator(llm_provider=None) -> ICrewOrchestrator:
        """Create CrewAI orchestrator with LLM provider."""
        return CrewOrchestrator(llm_provider=llm_provider)