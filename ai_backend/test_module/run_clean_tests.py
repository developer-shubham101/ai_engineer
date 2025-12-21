#!/usr/bin/env python3
"""
Clean test runner for the AI backend system.
Focuses on API-based tests that work with the current server.
"""

import sys
import importlib.util
from pathlib import Path

def run_test_file(test_file_path):
    """Run a single test file and return success status."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)
        
        # Add the tests directory to sys.path for relative imports
        tests_dir = str(Path(test_file_path).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        
        spec.loader.exec_module(test_module)
        
        # If the module has a main function or test function, call it
        if hasattr(test_module, 'main'):
            return test_module.main()
        elif hasattr(test_module, 'test_' + Path(test_file_path).stem.replace('test_', '')):
            test_func = getattr(test_module, 'test_' + Path(test_file_path).stem.replace('test_', ''))
            return test_func()
        else:
            print(f"✅ {test_file_path.name} loaded successfully")
            return True
            
    except Exception as e:
        print(f"❌ {test_file_path.name} failed: {e}")
        return False

def main():
    """Run all working tests."""
    
    print("🧪 AI Backend Test Suite")
    print("=" * 50)
    print("Server: http://127.0.0.1:8000")
    print("=" * 50)
    
    tests_dir = Path(__file__).parent
    
    # List of working tests (API-based, no old service imports)
    working_tests = [
        "test_live_endpoints.py",
        "test_temperature.py", 
        "test_conversation_context.py",
        "test_optimized_prompt.py",
        "test_embedding_simple.py",
        "test_rbac_endpoints.py",
        "test_api_metadata_validation.py"
    ]
    
    results = {}
    
    for test_file in working_tests:
        test_path = tests_dir / test_file
        
        if test_path.exists():
            print(f"\n🔄 Running {test_file}...")
            try:
                success = run_test_file(test_path)
                results[test_file] = success
            except Exception as e:
                print(f"❌ {test_file} execution failed: {e}")
                results[test_file] = False
        else:
            print(f"⚠️  {test_file} not found, skipping...")
            results[test_file] = None
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for test_file, result in results.items():
        if result is True:
            print(f"✅ {test_file}")
        elif result is False:
            print(f"❌ {test_file}")
        else:
            print(f"⚠️  {test_file} (skipped)")
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All available tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)