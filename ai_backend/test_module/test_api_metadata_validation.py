#!/usr/bin/env python3
"""
Test API metadata validation and role-based restrictions.

This script tests:
1. Department validation
2. Sensitivity validation
3. Role-based sensitivity restrictions
4. allowed_roles validation
5. Personal document owner_id requirement
"""
import requests
import json

from .constants import BASE_URL

def print_test(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

def login(username, password):
    """Login and get JWT token"""
    url = f"{BASE_URL}/api/auth/token"
    response = requests.post(url, json={"username": username, "password": password})
    if response.status_code == 200:
        return response.json().get("access_token")
    return None

def test_add_document(token, doc_data, expected_status=200):
    """Test adding a document"""
    url = f"{BASE_URL}/api/rag/documents/add"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=doc_data)
    
    print(f"Status: {response.status_code} (Expected: {expected_status})")
    if response.status_code == expected_status:
        print("✅ PASS")
    else:
        print(f"❌ FAIL - Response: {response.text[:200]}")
    
    return response.status_code == expected_status

def main():
    print("API Metadata Validation Test Suite")
    
    # Login as different users
    employee_token = login("employee", "emp123")
    manager_token = login("manager", "mgr123")
    hr_token = login("hr_manager", "hr123")
    admin_token = login("admin", "admin123")
    
    # Test 1: Invalid Department
    print_test("Invalid Department (should fail)")
    test_add_document(employee_token, {
        "source_name": "test_invalid_dept.md",
        "text": "Test content",
        "metadata": {
            "department": "InvalidDept",  # Invalid
            "sensitivity": "public_internal"
        }
    }, expected_status=400)
    
    # Test 2: Invalid Sensitivity
    print_test("Invalid Sensitivity Level (should fail)")
    test_add_document(employee_token, {
        "source_name": "test_invalid_sens.md",
        "text": "Test content",
        "metadata": {
            "department": "Engineering",
            "sensitivity": "super_secret"  # Invalid
        }
    }, expected_status=400)
    
    # Test 3: Employee trying to create highly_confidential (should fail)
    print_test("Employee Creating Highly Confidential Doc (should fail)")
    test_add_document(employee_token, {
        "source_name": "test_employee_confidential.md",
        "text": "Test content",
        "metadata": {
            "department": "Engineering",
            "sensitivity": "highly_confidential"  # Not allowed for Employee
        }
    }, expected_status=403)
    
    # Test 4: Manager creating department_confidential (should succeed)
    print_test("Manager Creating Department Confidential Doc (should succeed)")
    test_add_document(manager_token, {
        "source_name": "test_manager_dept_conf.md",
        "text": "Test content",
        "metadata": {
            "department": "Engineering",
            "sensitivity": "department_confidential"  # Allowed for Manager
        }
    }, expected_status=200)
    
    # Test 5: HR creating personal document without owner_id (should fail)
    print_test("Personal Doc Without owner_id (should fail)")
    test_add_document(hr_token, {
        "source_name": "test_personal_no_owner.md",
        "text": "Test content",
        "metadata": {
            "department": "HR",
            "sensitivity": "personal"
            # Missing owner_id
        }
    }, expected_status=400)
    
    # Test 6: HR creating personal document with owner_id (should succeed)
    print_test("Personal Doc With owner_id (should succeed)")
    test_add_document(hr_token, {
        "source_name": "test_personal_with_owner.md",
        "text": "Test content",
        "metadata": {
            "department": "HR",
            "sensitivity": "personal",
            "owner_id": "u_emp_1"  # Valid
        }
    }, expected_status=200)
    
    # Test 7: Invalid allowed_roles (should fail)
    print_test("Invalid allowed_roles (should fail)")
    test_add_document(hr_token, {
        "source_name": "test_invalid_roles.md",
        "text": "Test content",
        "metadata": {
            "department": "HR",
            "sensitivity": "role_confidential",
            "allowed_roles": ["InvalidRole", "Manager"]  # InvalidRole doesn't exist
        }
    }, expected_status=400)
    
    # Test 8: Valid allowed_roles (should succeed)
    print_test("Valid allowed_roles (should succeed)")
    test_add_document(hr_token, {
        "source_name": "test_valid_roles.md",
        "text": "Test content",
        "metadata": {
            "department": "HR",
            "sensitivity": "role_confidential",
            "allowed_roles": ["HR", "Manager"]  # Valid
        }
    }, expected_status=200)
    
    # Test 9: SuperAdmin creating highly_confidential (should succeed)
    print_test("SuperAdmin Creating Highly Confidential (should succeed)")
    test_add_document(admin_token, {
        "source_name": "test_admin_highly_conf.md",
        "text": "Test content",
        "metadata": {
            "department": "Legal",
            "sensitivity": "highly_confidential"  # Allowed for SuperAdmin
        }
    }, expected_status=200)
    
    print("\n" + "="*60)
    print("API Metadata Validation Tests Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
