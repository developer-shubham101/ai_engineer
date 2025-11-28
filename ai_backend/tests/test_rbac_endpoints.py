#!/usr/bin/env python3
"""
Test script for RBAC endpoints with new flexible role system.
"""

import requests
import json

from .constants import BASE_URL

def test_rbac_scenarios():
    """Test various RBAC scenarios"""
    
    print("=== RBAC ENDPOINT TESTS ===\n")
    
    # Test document creation with different sensitivity levels
    test_cases = [
        {
            "name": "Employee creates public document",
            "user": "employee",
            "doc": {
                "source_name": "Public Policy",
                "text": "This is public information",
                "metadata": {"sensitivity": "public_internal", "department": "Engineering"}
            },
            "expected": "SUCCESS"
        },
        {
            "name": "Employee tries super_confidential",
            "user": "employee", 
            "doc": {
                "source_name": "Top Secret",
                "text": "Classified information",
                "metadata": {"sensitivity": "super_confidential", "department": "Engineering"}
            },
            "expected": "FORBIDDEN - Level too low"
        },
        {
            "name": "Manager creates highly_confidential",
            "user": "manager",
            "doc": {
                "source_name": "Management Decision", 
                "text": "Strategic planning document",
                "metadata": {"sensitivity": "highly_confidential", "department": "Engineering"}
            },
            "expected": "SUCCESS"
        },
        {
            "name": "Document with role override (Admin+Employee only)",
            "user": "admin",
            "doc": {
                "source_name": "Special Project",
                "text": "Only admin and employees can see this",
                "metadata": {
                    "sensitivity": "highly_confidential",
                    "allowed_roles": ["SuperAdmin", "Employee"],
                    "department": "Engineering"
                }
            },
            "expected": "SUCCESS - Role override"
        }
    ]
    
    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"Expected: {case['expected']}")
        print(f"Document: {case['doc']['source_name']}")
        print(f"Sensitivity: {case['doc']['metadata']['sensitivity']}")
        if 'allowed_roles' in case['doc']['metadata']:
            print(f"Allowed Roles: {case['doc']['metadata']['allowed_roles']}")
        print("---")
    
    print("\n=== QUERY ACCESS TESTS ===")
    
    query_tests = [
        {
            "name": "Manager queries Admin+Employee doc",
            "user": "manager",
            "query": "What is the special project about?",
            "expected": "BLOCKED - Not in allowed_roles despite high level"
        },
        {
            "name": "Employee queries Admin+Employee doc", 
            "user": "employee",
            "query": "What is the special project about?",
            "expected": "SUCCESS - In allowed_roles"
        },
        {
            "name": "HR queries department doc",
            "user": "hr_manager",
            "query": "What are the HR policies?", 
            "expected": "SUCCESS - Department match + role level"
        }
    ]
    
    for test in query_tests:
        print(f"Test: {test['name']}")
        print(f"User: {test['user']}")
        print(f"Expected: {test['expected']}")
        print("---")

if __name__ == "__main__":
    test_rbac_scenarios()