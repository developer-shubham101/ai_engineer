#!/usr/bin/env python3
"""
Simple Embedding Model API Test
Quick test for embedding model upgrade system
"""

import requests
import json

BASE_URL = "http://localhost:5444"

def test_embedding_api():
    print("🧪 Simple Embedding Model Test")
    
    # 1. Get token
    auth_response = requests.post(f"{BASE_URL}/api/auth/token", 
        json={"username": "admin", "password": "admin123"})
    
    if auth_response.status_code != 200:
        print("❌ Auth failed")
        return
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Test embedding status
    status_response = requests.get(f"{BASE_URL}/api/rag/embedding/status", headers=headers)
    
    if status_response.status_code == 200:
        data = status_response.json()
        model_info = data.get("embedding_model", {})
        print(f"✅ Model: {model_info.get('model_key')}")
        print(f"✅ Loaded: {model_info.get('model_loaded')}")
        print(f"✅ Dimensions: {model_info.get('actual_dimensions')}")
    else:
        print("❌ Status check failed")
    
    # 3. Test query performance
    query_response = requests.post(f"{BASE_URL}/api/rag/local/query", 
        headers=headers,
        json={
            "question": "What is our company policy?",
            "top_k": 3,
            "use_llm": False
        })
    
    if query_response.status_code == 200:
        data = query_response.json()
        docs = data.get("retrieved", [])
        print(f"✅ Query returned {len(docs)} documents")
    else:
        print("❌ Query failed")
    
    print("🎉 Test completed")

if __name__ == "__main__":
    test_embedding_api()