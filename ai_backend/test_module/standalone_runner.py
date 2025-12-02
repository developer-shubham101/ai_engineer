"""Test runner for all container validation tests."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_authenticator import run_standalone_tests as test_auth
from test_user_manager import run_standalone_tests as test_user
from test_session_manager import run_standalone_tests as test_session
from test_vector_store import run_standalone_tests as test_vector
from test_rag_orchestrator import run_standalone_tests as test_rag


async def run_all_tests():
    """Run all validation tests in sequence."""
    print("=" * 60)
    print("CONTAINER VALIDATION SUITE")
    print("=" * 60)
    
    test_functions = [
        ("Authenticator", test_auth),
        ("User Manager", test_user),
        ("Session Manager", test_session),
        ("Vector Store", test_vector),
        ("RAG Orchestrator", test_rag),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'=' * 20} {test_name} {'=' * 20}")
            await test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name} tests failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print(f"[WARNING] {failed} test(s) failed")
    
    return failed == 0


async def run_specific_test(test_name: str):
    """Run a specific test by name."""
    test_map = {
        "auth": test_auth,
        "user": test_user,
        "session": test_session,
        "vector": test_vector,
        "rag": test_rag,
    }
    
    if test_name.lower() in test_map:
        await test_map[test_name.lower()]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_map.keys())}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        asyncio.run(run_specific_test(sys.argv[1]))
    else:
        # Run all tests
        asyncio.run(run_all_tests())