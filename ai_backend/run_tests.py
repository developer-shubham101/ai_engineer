#!/usr/bin/env python3
"""
Test execution script for container validation.
Demonstrates both old and new test approaches.
"""
import asyncio
import subprocess
import sys


async def run_original_validation():
    """Run the original (now improved) validation script."""
    print("🔄 Running Original Validation Script")
    print("=" * 50)
    
    try:
        from test_module.validate_container_full import main
        await main()
    except Exception as e:
        print(f"Original validation failed: {e}")


def run_pytest_tests():
    """Run the new pytest-based tests."""
    print("\n🧪 Running Pytest Test Suite")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_module/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Pytest execution failed: {e}")
        return False


async def run_standalone_tests():
    """Run the new standalone test runner."""
    print("\n🚀 Running Standalone Test Runner")
    print("=" * 50)
    
    try:
        from test_module.test_runner import run_all_tests
        success = await run_all_tests()
        return success
    except Exception as e:
        print(f"Standalone test runner failed: {e}")
        return False


async def main():
    """Run all test approaches."""
    print("🎯 Container Validation Test Suite")
    print("=" * 60)
    
    # Run original validation
    await run_original_validation()
    
    # Run new standalone tests
    standalone_success = await run_standalone_tests()
    
    # Run pytest tests (if available)
    pytest_success = run_pytest_tests()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Standalone Tests: {'✅ PASSED' if standalone_success else '❌ FAILED'}")
    print(f"Pytest Tests:     {'✅ PASSED' if pytest_success else '❌ FAILED'}")
    
    if standalone_success and pytest_success:
        print("\n🎉 ALL TEST SUITES PASSED!")
    else:
        print("\n⚠️  Some tests failed - check output above")


if __name__ == "__main__":
    asyncio.run(main())