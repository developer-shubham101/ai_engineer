#!/usr/bin/env python3
"""
Test script demonstrating flexible RBAC system.
Shows how level-based access works with specific role overrides.
"""

def test_rbac_logic():
    """Test the flexible RBAC logic"""
    
    # Role levels
    ROLE_LEVELS = {"SuperAdmin": 4, "Manager": 3, "HR": 2, "Employee": 1, "Guest": 0}
    SENSITIVITY_LEVELS = {"public_internal": 0, "department_confidential": 1, "role_confidential": 2, "highly_confidential": 3, "super_confidential": 4}
    
    def check_access(user_role, doc_sensitivity, allowed_roles=None, user_dept=None, doc_dept=None):
        user_level = ROLE_LEVELS.get(user_role, 0)
        
        # Specific role override
        if allowed_roles:
            return user_role in allowed_roles
        
        # Department check for dept_confidential
        if doc_sensitivity == "department_confidential" and user_dept == doc_dept:
            return True
            
        # Level-based access
        required_level = SENSITIVITY_LEVELS.get(doc_sensitivity, 0)
        return user_level >= required_level
    
    print("=== FLEXIBLE RBAC TEST CASES ===\n")
    
    # Test Case 1: Normal hierarchy (higher levels access lower)
    print("1. NORMAL HIERARCHY:")
    print(f"SuperAdmin → public_internal: {check_access('SuperAdmin', 'public_internal')}")
    print(f"Employee → highly_confidential: {check_access('Employee', 'highly_confidential')}")
    print(f"Manager → role_confidential: {check_access('Manager', 'role_confidential')}")
    
    # Test Case 2: Specific role override (your example)
    print("\n2. SPECIFIC ROLE OVERRIDE (Admin+Employee only):")
    allowed_roles = ["SuperAdmin", "Employee"]  # Only these can access
    print(f"SuperAdmin → doc: {check_access('SuperAdmin', 'highly_confidential', allowed_roles)}")
    print(f"Employee → doc: {check_access('Employee', 'highly_confidential', allowed_roles)}")
    print(f"Manager → doc: {check_access('Manager', 'highly_confidential', allowed_roles)}")  # Blocked!
    print(f"HR → doc: {check_access('HR', 'highly_confidential', allowed_roles)}")  # Blocked!
    
    # Test Case 3: Department-specific access
    print("\n3. DEPARTMENT ACCESS:")
    print(f"HR user → HR dept doc: {check_access('Employee', 'department_confidential', None, 'HR', 'HR')}")
    print(f"Eng user → HR dept doc: {check_access('Employee', 'department_confidential', None, 'Engineering', 'HR')}")
    
    print("\n=== USAGE EXAMPLES ===")
    print("Document metadata examples:")
    print('{"sensitivity": "highly_confidential", "allowed_roles": ["SuperAdmin", "Employee"]}')
    print('{"sensitivity": "department_confidential", "department": "HR"}')
    print('{"sensitivity": "super_confidential"}  # SuperAdmin only')

if __name__ == "__main__":
    test_rbac_logic()