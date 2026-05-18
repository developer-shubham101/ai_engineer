import sys
import os

# Add the parent directory to sys.path so we can import app modules
# Assuming we run this from ai_backend/
sys.path.append(os.getcwd())

try:
    from app.modules.agents.orchestrators.autogen_orchestrator import AutoGenOrchestrator
    print("Import successful")
    
    # Try to instantiate
    orch = AutoGenOrchestrator()
    print("Instantiation successful")
    
    # We won't run it because we don't know if llama-server is actually up, 
    # but successful instantiation confirms imports are correct.
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
