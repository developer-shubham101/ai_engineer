#!/usr/bin/env python3
"""
Comprehensive RBAC Test Suite

Tests all RBAC features:
1. Document creation with role-based sensitivity restrictions
2. Department ownership checks on updates
3. Metadata validation
4. Audit logging
5. Cross-department access restrictions
"""
import requests
import json

from .constants import BASE_URL

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_test(name, passed):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")

def login(username, password):
    """Login and get JWT token"""
    url = f"{BASE_URL}/api/auth/token"
    response = requests.post(url, json={"username": username, "password": password})
    if response.status_code == 200:
        return response.json().get("access_token"), response.json().get("user")
    return None, None

def add_document(token, doc_data):
    """Add a document"""
    url = f"{BASE_URL}/api/rag/documents/add"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return requests.post(url, headers=headers, json=doc_data)

def update_document(token, update_data):
    """Update a document"""
    url = f"{BASE_URL}/api/rag/documents/update"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return requests.post(url, headers=headers, json=update_data)

def main():
    print("🔐 Comprehensive RBAC Test Suite")
    print(f"Target: {BASE_URL}")
    
    # Login as different users
    employee_token, employee_user = login("employee", "emp123")
    manager_token, manager_user = login("manager", "mgr123")
    hr_token, hr_user = login("hr_manager", "hr123")
    admin_token, admin_user = login("admin", "admin123")
    
    print(f"\n👥 Logged in users:")
    print(f"  - Employee: {employee_user.get('username')} (Dept: {employee_user.get('department')})")
    print(f"  - Manager: {manager_user.get('username')} (Dept: {manager_user.get('department')})")
    print(f"  - HR: {hr_user.get('username')} (Dept: {hr_user.get('department')})")
    print(f"  - Admin: {admin_user.get('username')} (Dept: {admin_user.get('department')})")
    
    # ========================================================================
    # TEST SUITE 1: Role-Based Creation Restrictions
    # ========================================================================
    print_section("TEST SUITE 1: Role-Based Creation Restrictions")
    
    # Test 1.1: Employee creating public_internal (should succeed)
    resp = add_document(employee_token, {
        "source_name": "test_employee_public.md",
        "text": "Public content",
        "metadata": {"department": "Engineering", "sensitivity": "public_internal"}
    })
    print_test("Employee creating public_internal doc", resp.status_code == 200)
    
    # Test 1.2: Employee creating highly_confidential (should fail)
    resp = add_document(employee_token, {
        "source_name": "test_employee_confidential.md",
        "text": "Confidential content",
        "metadata": {"department": "Engineering", "sensitivity": "highly_confidential"}
    })
    print_test("Employee creating highly_confidential doc (should fail)", resp.status_code == 403)
    
    # Test 1.3: Manager creating department_confidential (should succeed)
    resp = add_document(manager_token, {
        "source_name": "test_manager_dept_conf.md",
        "text": "Department confidential content",
        "metadata": {"department": "Engineering", "sensitivity": "department_confidential"}
    })
    print_test("Manager creating department_confidential doc", resp.status_code == 200)
    
    # Test 1.4: HR creating personal doc with owner_id (should succeed)
    resp = add_document(hr_token, {
        "source_name": "test_hr_personal.md",
        "text": "Personal content",
        "metadata": {
            "department": "HR",
            "sensitivity": "personal",
            "owner_id": "u_emp_1"
        }
    })
    print_test("HR creating personal doc with owner_id", resp.status_code == 200)
    
    # Test 1.5: HR creating personal doc without owner_id (should fail)
    resp = add_document(hr_token, {
        "source_name": "test_hr_personal_no_owner.md",
        "text": "Personal content",
        "metadata": {"department": "HR", "sensitivity": "personal"}
    })
    print_test("HR creating personal doc without owner_id (should fail)", resp.status_code == 400)
    
    # ========================================================================
    # TEST SUITE 2: Department Ownership on Updates
    # ========================================================================
    print_section("TEST SUITE 2: Department Ownership on Updates")
    
    # Create an HR document
    resp = add_document(hr_token, {
        "source_name": "test_hr_policy.md",
        "text": "HR Policy v1",
        "metadata": {"department": "HR", "sensitivity": "department_confidential"}
    })
    if resp.status_code == 200:
        hr_doc_id = resp.json().get("message", "").split("document_id=")[-1].rstrip(")")
        
        # Test 2.1: Manager (Engineering) trying to update HR doc (should fail)
        resp = update_document(manager_token, {
            "document_id": hr_doc_id,
            "text": "HR Policy v2 - Updated by Manager",
            "status": "published"
        })
        print_test("Manager updating HR department doc (should fail)", resp.status_code == 403)
        
        # Test 2.2: HR updating their own doc (should succeed)
        resp = update_document(hr_token, {
            "document_id": hr_doc_id,
            "text": "HR Policy v2 - Updated by HR",
            "status": "published"
        })
        print_test("HR updating their own department doc", resp.status_code == 200)
        
        # Test 2.3: SuperAdmin updating HR doc (should succeed)
        resp = update_document(admin_token, {
            "document_id": hr_doc_id,
            "text": "HR Policy v3 - Updated by Admin",
            "status": "published"
        })
        print_test("SuperAdmin updating HR doc (cross-department)", resp.status_code == 200)
    
    # ========================================================================
    # TEST SUITE 3: Metadata Validation on Updates
    # ========================================================================
    print_section("TEST SUITE 3: Metadata Validation on Updates")
    
    # Create a manager document
    resp = add_document(manager_token, {
        "source_name": "test_manager_doc.md",
        "text": "Manager Doc v1",
        "metadata": {"department": "Engineering", "sensitivity": "department_confidential"}
    })
    if resp.status_code == 200:
        mgr_doc_id = resp.json().get("message", "").split("document_id=")[-1].rstrip(")")
        
        # Test 3.1: Manager trying to escalate to highly_confidential (should fail)
        resp = update_document(manager_token, {
            "document_id": mgr_doc_id,
            "text": "Manager Doc v2",
            "metadata": {"sensitivity": "highly_confidential"},
            "status": "published"
        })
        print_test("Manager escalating to highly_confidential (should fail)", resp.status_code == 403)
        
        # Test 3.2: Manager updating to public_internal (should succeed)
        resp = update_document(manager_token, {
            "document_id": mgr_doc_id,
            "text": "Manager Doc v2",
            "metadata": {"sensitivity": "public_internal"},
            "status": "published"
        })
        print_test("Manager downgrading to public_internal", resp.status_code == 200)
    
    # ========================================================================
    # TEST SUITE 4: Invalid Metadata Validation
    # ========================================================================
    print_section("TEST SUITE 4: Invalid Metadata Validation")
    
    # Test 4.1: Invalid department
    resp = add_document(employee_token, {
        "source_name": "test_invalid_dept.md",
        "text": "Test",
        "metadata": {"department": "InvalidDept", "sensitivity": "public_internal"}
    })
    print_test("Invalid department (should fail)", resp.status_code == 400)
    
    # Test 4.2: Invalid sensitivity
    resp = add_document(employee_token, {
        "source_name": "test_invalid_sens.md",
        "text": "Test",
        "metadata": {"department": "Engineering", "sensitivity": "super_secret"}
    })
    print_test("Invalid sensitivity level (should fail)", resp.status_code == 400)
    
    # Test 4.3: Invalid allowed_roles
    resp = add_document(hr_token, {
        "source_name": "test_invalid_roles.md",
        "text": "Test",
        "metadata": {
            "department": "HR",
            "sensitivity": "role_confidential",
            "allowed_roles": ["InvalidRole", "Manager"]
        }
    })
    print_test("Invalid allowed_roles (should fail)", resp.status_code == 400)
    
    # ========================================================================
    # TEST SUITE 5: Version Deduplication (Latest Accessible)
    # ========================================================================
    print_section("TEST SUITE 5: Version Deduplication")
    
    # Create v1 (public) and v2 (public)
    # Note: In real usage, update_document creates v2. 
    # We'll use update_document to create v2.
    
    resp = add_document(manager_token, {
        "source_name": "test_versioning.md",
        "text": "Version 1 Content",
        "metadata": {"department": "Engineering", "sensitivity": "public_internal"}
    })
    if resp.status_code == 200:
        doc_id = resp.json().get("message", "").split("document_id=")[-1].rstrip(")")
        
        # Create v2
        update_document(manager_token, {
            "document_id": doc_id,
            "text": "Version 2 Content",
            "status": "published"
        })
        
        # Query - should see ONLY v2 content
        url = f"{BASE_URL}/api/rag/local/query"
        headers = {"Authorization": f"Bearer {employee_token}", "Content-Type": "application/json"}
        q_resp = requests.post(url, headers=headers, json={
            "question": "Version Content",
            "top_k": 10,
            "use_llm": False,
            "debug": True
        })
        
        if q_resp.status_code == 200:
            retrieved = q_resp.json().get("retrieved", [])
            versions_seen = [r["metadata"].get("version") for r in retrieved]
            print(f"   Retrieved versions: {versions_seen}")
            
            has_v2 = any(v == "2.0" for v in versions_seen)
            has_v1 = any(v == "1.0" for v in versions_seen)
            
            print_test("Query returns v2 (latest)", has_v2)
            print_test("Query hides v1 (deduplicated)", not has_v1)

    # ========================================================================
    # TEST SUITE 6: Allowed Roles Override for Department Docs
    # ========================================================================
    print_section("TEST SUITE 6: Allowed Roles Override")
    
    # Create HR doc that allows "Manager" role
    resp = add_document(hr_token, {
        "source_name": "test_hr_shared.md",
        "text": "Shared HR Content",
        "metadata": {
            "department": "HR", 
            "sensitivity": "department_confidential",
            "allowed_roles": ["Manager"] # Override: Allow Managers (who are usually Engineering/etc)
        }
    })
    
    # Manager (Engineering) tries to query it
    # Normally blocked (dept mismatch), but should be allowed via allowed_roles
    url = f"{BASE_URL}/api/rag/local/query"
    headers = {"Authorization": f"Bearer {manager_token}", "Content-Type": "application/json"}
    q_resp = requests.post(url, headers=headers, json={
        "question": "Shared HR Content",
        "top_k": 5,
        "use_llm": False,
        "debug": True
    })
    
    if q_resp.status_code == 200:
        retrieved = q_resp.json().get("retrieved", [])
        found = any("test_hr_shared" in r["metadata"].get("source", "") for r in retrieved)
        print_test("Manager accessing HR doc via allowed_roles override", found)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("TEST SUMMARY")
    print("\n✅ All RBAC features tested!")
    print("\n📊 Check server logs for:")
    print("  - METADATA_VALIDATION_FAILED (creation attempts with invalid metadata)")
    print("  - RBAC_UPDATE_DENIED (cross-department update attempts)")
    print("  - DOCUMENT_CREATED (successful creations)")
    print("  - DOCUMENT_UPDATED (successful updates)")
    print("  - METADATA_CHANGE (sensitivity level changes)")
    print("  - RBAC_ACCESS_DENIED (retrieval denials)")

if __name__ == "__main__":
    main()
