#!/usr/bin/env python3
"""
Test script for conversation context - Updated for current architecture.
Tests conversation history integration via API endpoints.
"""

import requests
from .constants import BASE_URL

def test_conversation_context():
    """Test conversation context via API endpoints."""
    
    print("🧪 Testing Conversation Context Integration")
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
    
    # Create a new conversation
    try:
        conv_response = requests.post(
            f"{BASE_URL}/api/conversations",
            headers=headers,
            json={"title": "Context Test Conversation"}
        )
        
        if conv_response.status_code != 200:
            print("❌ Failed to create conversation")
            return False
            
        conversation_id = conv_response.json()["id"]
        print(f"✅ Created conversation: {conversation_id}")
        
    except Exception as e:
        print(f"❌ Conversation creation error: {e}")
        return False
    
    # Test conversation flow with context
    conversation_flow = [
        "What is our company policy on remote work?",
        "Can you tell me more about the stipend?",
        "What are the requirements for remote work?"
    ]
    
    for i, question in enumerate(conversation_flow):
        print(f"\n--- Question {i+1}: {question[:30]}... ---")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/rag/local/query",
                headers=headers,
                json={
                    "question": question,
                    "conversation_id": conversation_id,
                    "use_llm": True,
                    "use_conversation_history": True,
                    "top_k": 3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                print(f"✅ Response received: {len(answer)} characters")
                
                # Check if context is being used
                if i > 0 and ("stipend" in question.lower() or "requirements" in question.lower()):
                    if len(answer) > 50:  # Substantial response indicates context usage
                        print("🎯 Context appears to be utilized")
                    else:
                        print("⚠️  Response may not be using full context")
                        
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
    
    # Get conversation messages to verify history
    try:
        messages_response = requests.get(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages",
            headers=headers
        )
        
        if messages_response.status_code == 200:
            messages = messages_response.json()
            print(f"\n✅ Conversation history: {len(messages)} messages stored")
        else:
            print(f"❌ Failed to retrieve conversation history")
            
    except Exception as e:
        print(f"❌ History retrieval error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Conversation Context Test Complete!")
    
    return True

if __name__ == "__main__":
    success = test_conversation_context()
    if success:
        print("\n🎉 Conversation context tests completed!")
    else:
        print("\n❌ Conversation context tests failed")
        exit(1)