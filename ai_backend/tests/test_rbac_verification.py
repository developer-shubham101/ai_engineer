#!/usr/bin/env python3
"""
Test RBAC (Role-Based Access Control) functionality.

This script tests:
1. Metadata loading from .meta.json files
2. RBAC filtering for different roles
3. Audit logging for blocked access
4. Public summaries for restricted content
"""
import requests
import json
import sys

from .constants import BASE_URL

# Test users with different roles
TEST_USERS = [
    {"username": "admin", "password": "admin123", "expected_role": "SuperAdmin"},
    {"username": "hr_manager", "password": "hr123", "expected_role": "HR"},
    {"username": "manager", "password": "mgr123", "expected_role": "Manager"},
    {"username": "employee", "password": "emp123", "expected_role": "Employee"},
    {"username": "guest", "password": "guest123", "expected_role": "Guest"},
]

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def login(username, password):
    """Login and get JWT token"""
    url = f"{BASE_URL}/api/auth/token"
    payload = {"username": username, "password": password}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("access_token"), data.get("user")
    else:
        print(f"❌ Login failed for {username}: {response.text}")
        return None, None

def test_query(token, user_info, question):
    """Test RAG query with specific user"""
    url = f"{BASE_URL}/api/rag/local/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "question": question,
        "top_k": 5,
        "use_llm": False,
        "debug": True
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"❌ Query failed: {response.text}")
        return None

def test_rbac_for_user(user_config):
    """Test RBAC for a specific user"""
    print_header(f"Testing: {user_config['username']} ({user_config['expected_role']})")
    
    # Login
    token, user_info = login(user_config["username"], user_config["password"])
    if not token:
        return
    
    print(f"✅ Logged in as: {user_info.get('username')} (Role: {user_info.get('role')}, Dept: {user_info.get('department')})")
    
    # Test 1: Query for HR policy (department_confidential)
    print("\n📝 Test 1: Querying HR Leave Policy (department_confidential)")
    result = test_query(token, user_info, "What is the company leave policy?")
    if result:
        answer = result.get("answer", "")
        retrieved = result.get("retrieved", [])
        
        print(f"   Retrieved {len(retrieved)} documents")
        if "permission" in answer.lower() or "access" in answer.lower():
            print(f"   Access denied (expected for non-HR roles)")
            print(f"   Answer: {answer[:100]}...")
        else:
            print(f"   ✅ Access granted")
            if retrieved:
                for doc in retrieved[:2]:
                    meta = doc.get("metadata", {})
                    print(f"   - Source: {meta.get('source_name', 'unknown')} (Sensitivity: {meta.get('sensitivity', 'unknown')})")
    
    # Test 2: Query for public policy (public_internal)
    print("\n📝 Test 2: Querying Remote Work Policy (public_internal)")
    result = test_query(token, user_info, "What is the remote work policy?")
    if result:
        answer = result.get("answer", "")
        retrieved = result.get("retrieved", [])
        
        print(f"   Retrieved {len(retrieved)} documents")
        if "permission" in answer.lower():
            print(f"   Unexpected access denial for public_internal content")
        else:
            print(f"   ✅ Access granted (expected for all roles)")
            if retrieved:
                for doc in retrieved[:2]:
                    meta = doc.get("metadata", {})
                    print(f"   - Source: {meta.get('source_name', 'unknown')} (Sensitivity: {meta.get('sensitivity', 'unknown')})")

def main():
    print("🔐 RBAC Verification Script")
    print(f"Target: {BASE_URL}")
    
    # Test each user
    for user_config in TEST_USERS:
        try:
            test_rbac_for_user(user_config)
        except Exception as e:
            print(f"❌ Error testing {user_config['username']}: {e}")
    
    print_header("RBAC Testing Complete")
    print("\n📊 Summary:")
    print("- Check server logs for RBAC_ACCESS_DENIED audit entries")
    print("- Verify that HR users can access department_confidential docs")
    print("- Verify that Employees/Guests see permission denied messages")
    print("- Verify that public_internal docs are accessible to all")

if __name__ == "__main__":
    main()
