#!/usr/bin/env python3
"""
Live endpoint testing for personalized RAG system.
Tests against running server at http://192.168.1.2:8000
"""

import time

import requests

from .constants import BASE_URL


def test_authentication():
    """Test user authentication and token generation"""
    print("=== AUTHENTICATION TESTS ===")

    # Test login for different user types
    users = [
        {"username": "admin", "password": "admin123", "expected_role": "SuperAdmin"},
        {"username": "employee", "password": "emp123", "expected_role": "Employee"},
        {"username": "guest", "password": "guest123", "expected_role": "Guest"}
    ]

    tokens = {}

    for user in users:
        try:
            response = requests.post(f"{BASE_URL}/api/auth/token", json={
                "username": user["username"],
                "password": user["password"]
            })

            if response.status_code == 200:
                data = response.json()
                tokens[user["username"]] = data["access_token"]
                print(f"✅ {user['username']} login: {data['user']['role']}")
            else:
                print(f"❌ {user['username']} login failed: {response.status_code}")

        except Exception as e:
            print(f"❌ {user['username']} login error: {e}")

    return tokens


def test_guest_onboarding_flow(tokens):
    """Test guest user onboarding flow with sequential questions"""
    print("\n=== GUEST ONBOARDING FLOW TEST ===")

    if "guest" not in tokens:
        print("❌ No guest token available")
        return

    headers = {"Authorization": f"Bearer {tokens['guest']}"}

    # Onboarding sequence based on onboarding_fields.json
    onboarding_sequence = [
        {"question": "Hi I want to connect with HR", "expected_response": "What is your name?"},
        {"question": "John Smith", "expected_response": "What is your gender?"},
        {"question": "Male", "expected_response": "What is your job role?"},
        {"question": "Python Developer", "expected_response": "What is your age?"},
        {"question": "28", "expected_response": "Where are you located?"},
        {"question": "New York", "expected_response": "Which department are you trying to reach?"},
        {"question": "HR", "expected_response": "Thank you! Your details have been saved."},
        {"question": "Are there any Python developer openings?", "expected_response": "Profile-based job matching"}
    ]

    print("Testing guest onboarding sequence...")

    for i, step in enumerate(onboarding_sequence):
        print(f"\n--- Step {i + 1}: {step['question'][:30]}... ---")

        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/google/query",
                headers=headers,
                json={
                    "question": step["question"],
                    "use_llm": True,
                    "top_k": 2
                }
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                print(f"✅ Response: {answer[:100]}...")
                print(f"Expected: {step['expected_response']}")

                # Check if we're in onboarding vs normal chat
                if i < 7:  # Onboarding phase
                    if any(keyword in answer.lower() for keyword in ['what is', 'which department', 'thank you']):
                        print("🎯 Onboarding question detected")
                    else:
                        print("Expected onboarding question")
                else:  # Post-onboarding
                    if len(answer) > 50:  # Substantial response
                        print("🎯 Personalized response with profile context")
                    else:
                        print("Expected detailed personalized response")

            else:
                print(f"❌ Request failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")

        # Small delay between requests
        time.sleep(0.5)


def test_personalized_queries(tokens):
    """Test personalized RAG queries with different user contexts"""
    print("\n=== PERSONALIZED QUERY TESTS ===")

    test_cases = [
        {
            "name": "Employee - Career Growth",
            "user": "employee",
            "query": "What career advancement opportunities are available for me?",
            "provider": "google",
            "expected": "Role-specific career guidance"
        },
        {
            "name": "Admin - System Query",
            "user": "admin",
            "query": "Show me highly confidential company policies",
            "provider": "google",
            "expected": "Access to restricted content"
        }
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")

        if case["user"] not in tokens:
            print(f"❌ No token for {case['user']}")
            continue

        headers = {"Authorization": f"Bearer {tokens[case['user']]}"}

        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/{case['provider']}/query",
                headers=headers,
                json={
                    "question": case["query"],
                    "use_llm": True,
                    "top_k": 3
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query successful")
                print(f"Answer: {data.get('answer', 'No answer')[:200]}...")
                print(f"Retrieved docs: {len(data.get('retrieved', []))}")
                print(f"Expected: {case['expected']}")
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Query error: {e}")


def test_rbac_enforcement(tokens):
    """Test RBAC filtering with different sensitivity levels"""
    print("\n=== RBAC ENFORCEMENT TESTS ===")

    # First, add test documents with different sensitivity levels
    if "admin" in tokens:
        admin_headers = {"Authorization": f"Bearer {tokens['admin']}"}

        test_docs = [
            {
                "source_name": "Public Company Policy",
                "text": "This is public information about our company values and mission.",
                "metadata": {"sensitivity": "public_internal", "department": "General"}
            },
            {
                "source_name": "HR Confidential Policy",
                "text": "Confidential HR procedures for employee management and reviews.",
                "metadata": {"sensitivity": "role_confidential", "department": "HR"}
            },
            {
                "source_name": "Admin Only Document",
                "text": "Super confidential strategic information for executives only.",
                "metadata": {"sensitivity": "super_confidential", "department": "Executive"}
            }
        ]

        print("Adding test documents...")
        for doc in test_docs:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/rag/documents/add",
                    headers=admin_headers,
                    json=doc
                )
                if response.status_code == 200:
                    print(f"✅ Added: {doc['source_name']}")
                else:
                    print(f"❌ Failed to add: {doc['source_name']}")
            except Exception as e:
                print(f"❌ Error adding doc: {e}")

    # Test access with different user roles
    rbac_tests = [
        {
            "user": "guest",
            "query": "What are the company policies?",
            "expected": "Only public content visible"
        },
        {
            "user": "employee",
            "query": "Show me HR procedures",
            "expected": "Limited access to HR content"
        },
        {
            "user": "admin",
            "query": "Show me all confidential information",
            "expected": "Full access to all content"
        }
    ]

    for test in rbac_tests:
        print(f"\n--- RBAC Test: {test['user']} ---")

        if test["user"] not in tokens:
            print(f"❌ No token for {test['user']}")
            continue

        headers = {"Authorization": f"Bearer {tokens[test['user']]}"}

        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/google/query",
                headers=headers,
                json={
                    "question": test["query"],
                    "use_llm": True,
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ RBAC Query successful")
                print(f"Retrieved docs: {len(data.get('retrieved', []))}")
                print(f"Answer length: {len(data.get('answer', ''))}")
                print(f"Expected: {test['expected']}")

                # Check if response indicates filtered content
                if "permission" in data.get('answer', '').lower():
                    print("Access restriction detected in response")

            else:
                print(f"❌ RBAC Query failed: {response.status_code}")

        except Exception as e:
            print(f"❌ RBAC Query error: {e}")


def test_multi_provider_support(tokens):
    """Test different LLM providers"""
    print("\n=== MULTI-PROVIDER TESTS ===")

    providers = ["google"]  # ["local", "google", "gpt", "hf"]
    query = "What is our company about?"

    if "employee" in tokens:
        headers = {"Authorization": f"Bearer {tokens['employee']}"}

        for provider in providers:
            print(f"\n--- Testing {provider.upper()} Provider ---")

            try:
                response = requests.post(
                    f"{BASE_URL}/api/rag/{provider}/query",
                    headers=headers,
                    json={
                        "question": query,
                        "use_llm": True,
                        "top_k": 2
                    },
                    timeout=30  # Some providers might be slower
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {provider} provider working")
                    print(f"Answer preview: {data.get('answer', 'No answer')[:100]}...")
                else:
                    print(f"❌ {provider} provider failed: {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"⏰ {provider} provider timeout (might need API keys)")
            except Exception as e:
                print(f"❌ {provider} provider error: {e}")


def main():
    """Run all endpoint tests"""
    print("🚀 LIVE ENDPOINT TESTING")
    print(f"Server: {BASE_URL}")
    print("=" * 50)

    # Test authentication first
    tokens = test_authentication()

    if not tokens:
        print("❌ No authentication tokens obtained. Check server status.")
        return

    # Test guest onboarding flow first
    test_guest_onboarding_flow(tokens)

    # Run personalized query tests
    test_personalized_queries(tokens)

    # Test RBAC enforcement
    test_rbac_enforcement(tokens)

    # Test multi-provider support
    test_multi_provider_support(tokens)

    print("\n" + "=" * 50)
    print("🎯 TESTING COMPLETE")
    print("Check results above for any failures or issues.")


if __name__ == "__main__":
    main()
