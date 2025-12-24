"""Factory for CrewAI components."""

from typing import Optional
from .interfaces import ICrewOrchestrator
from .orchestrator import CrewOrchestrator


class CrewOrchestratorFactory:
    """Factory for creating CrewAI orchestrators."""
    
    @staticmethod
    def create_orchestrator() -> ICrewOrchestrator:
        """Create CrewAI orchestrator."""
        return CrewOrchestrator()