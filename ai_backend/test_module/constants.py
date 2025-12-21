"""Test constants and credentials.

This module provides centralized test configuration and user credentials
that match the seeded users in user_manager.py.
"""

# Base URL for API testing
BASE_URL = "http://localhost:8000"

# Test user credentials (matching user_manager.py DUMMY_USERS)
TEST_USERS = {
    "admin": {
        "username": "admin",
        "password": "admin123",
        "user_id": "u_admin_1",
        "role": "SuperAdmin",
        "department": "Executive"
    },
    "hr_manager": {
        "username": "hr_manager",
        "password": "hr123",
        "user_id": "u_hr_1",
        "role": "HR",
        "department": "HR"
    },
    "manager": {
        "username": "manager",
        "password": "mgr123",
        "user_id": "u_mgr_1",
        "role": "Manager",
        "department": "Engineering"
    },
    "employee": {
        "username": "employee",
        "password": "emp123",
        "user_id": "u_emp_1",
        "role": "Employee",
        "department": "Engineering"
    },
    "guest": {
        "username": "guest",
        "password": "guest123",
        "user_id": "u_guest_1",
        "role": "Guest",
        "department": "General"
    }
}

# Convenience accessors for common test scenarios
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin123"}
HR_CREDENTIALS = {"username": "hr_manager", "password": "hr123"}
MANAGER_CREDENTIALS = {"username": "manager", "password": "mgr123"}
EMPLOYEE_CREDENTIALS = {"username": "employee", "password": "emp123"}
GUEST_CREDENTIALS = {"username": "guest", "password": "guest123"}

# User IDs for direct user lookups
ADMIN_USER_ID = "u_admin_1"
HR_USER_ID = "u_hr_1"
MANAGER_USER_ID = "u_mgr_1"
EMPLOYEE_USER_ID = "u_emp_1"
GUEST_USER_ID = "u_guest_1"