"""
Test script for JWT authentication system.
Tests token generation, authentication, and role-based access control.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.user_service import init_user_db, authenticate_user
from app.services.auth import create_access_token, verify_token


async def test_authentication():
    print("=" * 60)
    print("Testing JWT Authentication System")
    print("=" * 60)
    
    # 1. Initialize database
    print("\n1. Initializing user database...")
    try:
        init_user_db(reset_on_start=True)  # Reset for testing
        print("[PASS] Database initialized successfully")
    except Exception as e:
        print(f"[FAIL] Database initialization failed: {e}")
        return
    
    # 2. Test authentication with valid credentials
    print("\n2. Testing authentication with valid credentials...")
    test_users = [
        ("admin", "admin123", "SuperAdmin"),
        ("hr_manager", "hr123", "HR"),
        ("manager", "mgr123", "Manager"),
        ("employee", "emp123", "Employee"),
    ]
    
    for username, password, expected_role in test_users:
        user_data = authenticate_user(username, password)
        if user_data and user_data["role"] == expected_role:
            print(f"[PASS] {username}: authenticated as {expected_role}")
        else:
            print(f"[FAIL] {username}: authentication failed")
    
    # 3. Test authentication with invalid credentials
    print("\n3. Testing authentication with invalid credentials...")
    invalid_user = authenticate_user("admin", "wrongpassword")
    if invalid_user is None:
        print("[PASS] Invalid password correctly rejected")
    else:
        print("[FAIL] Invalid password was accepted (security issue!)")
    
    # 4. Test token generation
    print("\n4. Testing JWT token generation...")
    user_data = authenticate_user("admin", "admin123")
    if user_data:
        token = create_access_token(user_data)
        print(f"[PASS] Token generated: {token[:50]}...")
        
        # 5. Test token verification
        print("\n5. Testing token verification...")
        decoded = verify_token(token)
        if decoded and decoded["user_id"] == user_data["user_id"]:
            print(f"[PASS] Token verified successfully")
            print(f"  - User ID: {decoded['user_id']}")
            print(f"  - Role: {decoded['role']}")
            print(f"  - Department: {decoded['department']}")
        else:
            print("[FAIL] Token verification failed")
    
    # 6. Test invalid token
    print("\n6. Testing invalid token...")
    invalid_token = "invalid.token.here"
    decoded = verify_token(invalid_token)
    if decoded is None:
        print("[PASS] Invalid token correctly rejected")
    else:
        print("[FAIL] Invalid token was accepted (security issue!)")
    
    print("\n" + "=" * 60)
    print("Authentication System Test Complete!")
    print("=" * 60)
    
    print("\nDummy User Credentials:")
    print("-" * 60)
    for username, password, role in test_users:
        print(f"  Username: {username:15} Password: {password:15} Role: {role}")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(test_authentication())
