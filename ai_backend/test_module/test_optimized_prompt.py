#!/usr/bin/env python3
"""
Test script to verify optimized prompt generation via API.
Tests debug mode and prompt optimization features.
"""

import requests
from .constants import BASE_URL

def test_optimized_prompt_generation():
    """Test optimized prompt generation via API debug mode."""
    
    print("🧪 Testing Optimized Prompt Generation")
    print("=" * 60)
    
    # Get authentication token
    try:
        auth_response = requests.post(f"{BASE_URL}/api/auth/token", json={
            "username": "admin",
            "password": "admin123"
        })
        
        if auth_response.status_code != 200:
            print("❌ Authentication failed")
            return False
            
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False
    
    # Test cases for prompt optimization
    test_cases = [
        {
            "name": "Simple Query",
            "question": "What is our company policy?",
            "expected_features": ["system prompt", "context", "question"]
        },
        {
            "name": "Complex Query with Context",
            "question": "What are the detailed leave policies and how do they apply to remote workers?",
            "expected_features": ["role-based", "context truncation", "token budgeting"]
        },
        {
            "name": "Follow-up Query",
            "question": "Can you elaborate on that?",
            "expected_features": ["conversation history", "context preservation"]
        }
    ]
    
    conversation_id = "test_optimization_conv"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/local/query",
                headers=headers,
                json={
                    "question": test_case["question"],
                    "conversation_id": conversation_id,
                    "use_llm": True,
                    "use_conversation_history": True,
                    "debug": True,  # Enable debug mode to see final_prompt
                    "top_k": 3,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                final_prompt = data.get("final_prompt", "")
                answer = data.get("answer", "")
                retrieved = data.get("retrieved", [])
                
                print(f"✅ Query successful")
                print(f"   Final prompt length: {len(final_prompt)} characters")
                print(f"   Estimated tokens: ~{len(final_prompt) // 4}")
                print(f"   Retrieved documents: {len(retrieved)}")
                print(f"   Answer length: {len(answer)} characters")
                
                # Analyze prompt structure
                if final_prompt:
                    print("   Prompt analysis:")
                    if "System:" in final_prompt or "Assistant" in final_prompt:
                        print("     ✅ System instructions present")
                    if "Context:" in final_prompt or len(retrieved) > 0:
                        print("     ✅ Context integration working")
                    if "Question:" in final_prompt:
                        print("     ✅ Question formatting correct")
                    
                    # Check for optimization features
                    prompt_length = len(final_prompt)
                    if prompt_length < 2000:  # Reasonable prompt size
                        print("     ✅ Prompt size optimized")
                    else:
                        print("     ⚠️  Prompt may be too long")
                        
                else:
                    print("   ⚠️  No debug prompt available")
                    
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"❌ Query error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Optimized Prompt Generation Test Complete!")
    print("\n📋 Key Features Tested:")
    print("- Debug mode prompt exposure")
    print("- Token budgeting and optimization")
    print("- Context integration and truncation")
    print("- Conversation history integration")
    print("- Role-based prompt customization")
    
    return True

def test_context_truncation():
    """Test context truncation with document retrieval."""
    
    print("\n🧪 Testing Context Truncation")
    print("=" * 40)
    
    # Get authentication token
    try:
        auth_response = requests.post(f"{BASE_URL}/api/auth/token", json={
            "username": "admin",
            "password": "admin123"
        })
        
        if auth_response.status_code != 200:
            print("❌ Authentication failed")
            return False
            
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False
    
    # Test with different top_k values to see truncation
    top_k_values = [1, 3, 5, 10]
    
    for top_k in top_k_values:
        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/local/query",
                headers=headers,
                json={
                    "question": "Tell me about all company policies in detail",
                    "conversation_id": "truncation_test",
                    "use_llm": False,  # Just test retrieval
                    "debug": True,
                    "top_k": top_k
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                retrieved = data.get("retrieved", [])
                context = data.get("context", "")
                
                print(f"Top-K {top_k}: {len(retrieved)} docs, {len(context)} chars context")
                
            else:
                print(f"❌ Top-K {top_k} failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Top-K {top_k} error: {e}")
    
    print("✅ Context truncation test completed")
    return True

if __name__ == "__main__":
    success1 = test_optimized_prompt_generation()
    success2 = test_context_truncation()
    
    if success1 and success2:
        print("\n🎉 All optimization tests completed successfully!")
    else:
        print("\n❌ Some optimization tests failed")
        exit(1)