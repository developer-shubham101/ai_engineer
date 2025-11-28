#!/usr/bin/env python3
"""
Test script to debug prompt generation in the RAG system.
"""
import requests
import json

from .constants import BASE_URL

# API endpoint
url = f"{BASE_URL}/api/rag/local/query"

# Headers
headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidV9hZG1pbl8xIiwidXNlcm5hbWUiOiJhZG1pbiIsInJvbGUiOiJTdXBlckFkbWluIiwiZGVwYXJ0bWVudCI6IkV4ZWN1dGl2ZSIsInNlc3Npb25faWQiOiJzZXNzXzJhNjE3OWQwZjU1MTRlYzY5Y2QxNjc4NDM1ZDRlYTM1IiwiZXhwIjoxNzY0MzU0NTI5LCJpYXQiOjE3NjQyNjgxMjl9.eJamHstgVm57hGm-HOVo0_gY188Laf9dn7oFJQ7_5w0",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "question": "Hi",
    "top_k": 3,
    "use_llm": False,  # Set to False as requested
    "max_tokens": 256,
    "category": "string",
    "debug": False,
    "local_llm_model": "string"
}

try:
    print("Sending request to:", url)
    print("Payload:", json.dumps(payload, indent=2))
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error Response:")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")