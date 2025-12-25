"""Orchestrators package for different agent implementations."""

from .custom.custom_orchestrator import CustomOrchestrator

try:
    from .autogen.autogen_orchestrator import AutoGenOrchestrator
    AUTOGEN_AVAILABLE = True
except ImportError:
    AutoGenOrchestrator = None
    AUTOGEN_AVAILABLE = False

__all__ = ["CustomOrchestrator", "AutoGenOrchestrator", "AUTOGEN_AVAILABLE"]