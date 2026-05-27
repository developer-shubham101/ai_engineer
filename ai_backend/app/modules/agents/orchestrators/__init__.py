"""Orchestrators package for different agent implementations."""
from .custom.custom_orchestrator import CustomOrchestrator

try:
    from .autogen.autogen_orchestrator import AutoGenOrchestrator
    AUTOGEN_AVAILABLE = True
except ImportError:
    AutoGenOrchestrator = None
    AUTOGEN_AVAILABLE = False

try:
    from .mcp.mcp_orchestrator import MCPOrchestrator
    MCP_AVAILABLE = True
except ImportError:
    MCPOrchestrator = None
    MCP_AVAILABLE = False

try:
    from .crewai.crewai_orchestrator import CrewAIOrchestrator
    CREWAI_AVAILABLE = True
except ImportError:
    CrewAIOrchestrator = None
    CREWAI_AVAILABLE = False

__all__ = [
    "CustomOrchestrator",
    "AutoGenOrchestrator", "AUTOGEN_AVAILABLE",
    "MCPOrchestrator", "MCP_AVAILABLE",
    "CrewAIOrchestrator", "CREWAI_AVAILABLE",
]