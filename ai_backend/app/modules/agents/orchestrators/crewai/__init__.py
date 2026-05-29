"""CrewAI module for multi-agent workflows."""
from __future__ import annotations

from .orchestrator import CrewAIOrchestrator
from .interfaces import ICrewOrchestrator, CrewRequest, CrewResponse

__all__ = ["CrewAIOrchestrator", "ICrewOrchestrator", "CrewRequest", "CrewResponse"]