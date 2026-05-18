import sys
import os

# Set up python path
sys.path.append(os.getcwd())

from app.modules.agents.factories import AgentOrchestratorFactory
from app.modules.agents.orchestrators.custom.custom_orchestrator import CustomOrchestrator
try:
    from app.modules.agents.orchestrators.autogen.autogen_orchestrator import AutoGenOrchestrator
except ImportError:
    AutoGenOrchestrator = None

def test_factories():
    print("Testing Factory...")
    
    # Test Custom
    orch_custom = AgentOrchestratorFactory.create_orchestrator(orchestrator_type="custom")
    print(f"Custom Type: {type(orch_custom)}")
    assert isinstance(orch_custom, CustomOrchestrator)
    print("Custom Orchestrator created successfully.")

    # Test AutoGen
    if AutoGenOrchestrator:
        orch_autogen = AgentOrchestratorFactory.create_orchestrator(orchestrator_type="autogen")
        print(f"AutoGen Type: {type(orch_autogen)}")
        assert isinstance(orch_autogen, AutoGenOrchestrator)
        print("AutoGen Orchestrator created successfully.")
    else:
        print("AutoGen not available, skipping test.")

if __name__ == "__main__":
    try:
        test_factories()
        print("Verification PASSED")
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
