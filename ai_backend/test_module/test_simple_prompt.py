#!/usr/bin/env python3
"""
Simple RAG Prompt Analysis Tool - Windows Compatible
Tests the complete prompt construction flow without Unicode characters.
"""

import requests
import json

# from .constants import BASE_URL # Removed relative import
BASE_URL = "http://localhost:8000" # Directly define BASE_URL

ADMIN_CREDENTIALS = {"username": "admin", "password": "admin123"}

def test_prompt_construction():
    """Test prompt construction with different scenarios."""
    
    # Authenticate
    session = requests.Session()
    auth_response = session.post(f"{BASE_URL}/api/auth/token", json=ADMIN_CREDENTIALS)
    
    if auth_response.status_code != 200:
        print(f"Authentication failed: {auth_response.status_code}")
        return
    
    token = auth_response.json()["access_token"]
    session.headers.update({"Authorization": f"Bearer {token}"})
    print("Authentication successful")
    
    print("\n" + "="*80)
    print("RAG PROMPT CONSTRUCTION ANALYSIS")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        {
            "name": "HR Policy Query",
            "query": "What is our leave policy?",
            "category": "HR"
        },
        {
            "name": "Technical Query", 
            "query": "How do I reset passwords?",
            "category": "IT"
        },
        {
            "name": "Sensitive Data Query",
            "query": "Show me salary information",
            "category": "HR"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        
        payload = {
            "question": scenario["query"],
            "top_k": 5,
            "use_llm": False,  # Don't call LLM, just see retrieval
            "debug": True
        }
        
        if scenario.get("category"):
            payload["category"] = scenario["category"]
        
        response = session.post(f"{BASE_URL}/api/rag/local/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            retrieved = data.get("retrieved", [])
            filtered_count = data.get("filtered_out_count", 0)
            context = data.get("context") or ""
            
            print(f"  Results:")
            print(f"    Retrieved Documents: {len(retrieved)}")
            print(f"    Filtered by RBAC: {filtered_count}")
            print(f"    Context Length: {len(context)} chars")
            
            # Show document metadata
            if retrieved:
                print(f"  Document Analysis:")
                for j, doc in enumerate(retrieved[:2]):
                    meta = doc.get("metadata", {})
                    dept = meta.get("department", "Unknown")
                    sens = meta.get("sensitivity", "Unknown")
                    source = meta.get("source", "Unknown")[:40]
                    print(f"    Doc {j+1}: {dept}/{sens} - {source}")
            
            # Check RBAC effectiveness
            if filtered_count > 0:
                total_found = len(retrieved) + filtered_count
                filter_ratio = (filtered_count / total_found) * 100
                print(f"  RBAC Filter Effectiveness: {filter_ratio:.1f}% blocked")
        else:
            print(f"  ERROR: {response.status_code}")

def test_with_llm():
    """Test with LLM enabled to see final prompt in logs."""
    
    session = requests.Session()
    auth_response = session.post(f"{BASE_URL}/api/auth/token", json=ADMIN_CREDENTIALS)
    
    if auth_response.status_code != 200:
        print("Authentication failed")
        return
    
    token = auth_response.json()["access_token"]
    session.headers.update({"Authorization": f"Bearer {token}"})
    
    print("\n" + "="*80)
    print("LLM PROMPT CONSTRUCTION TEST")
    print("="*80)
    
    test_query = "What are our company policies?"
    
    print(f"Testing with LLM enabled:")
    print(f"Query: {test_query}")
    
    response = session.post(
        f"{BASE_URL}/api/rag/local/query",
        json={
            "question": test_query,
            "top_k": 3,
            "use_llm": True,
            "max_tokens": 100
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data.get("answer", "No answer")
        print(f"LLM Response: {answer[:200]}...")
        print("\nNOTE: Check application logs for 'LLM_FINAL_PROMPT' entries")
        print("This shows the complete prompt sent to the LLM including:")
        print("- Tone guidance")
        print("- User profile context") 
        print("- Chat history")
        print("- Role/department context")
        print("- Retrieved document context")
    else:
        print(f"ERROR: {response.status_code}")

def show_prompt_flow():
    """Show the prompt construction flow."""
    
    print("\n" + "="*80)
    print("PROMPT CONSTRUCTION FLOW")
    print("="*80)
    
    flow_steps = [
        "1. USER QUERY -> API Endpoint",
        "2. Extract user context (role, dept, profile)",
        "3. Embed query with MiniLM",
        "4. Retrieve documents from ChromaDB", 
        "5. Apply RBAC filtering",
        "6. Build enhanced prompt:",
        "   - Base system prompt",
        "   - Tone guidance (sentiment)",
        "   - User profile context", 
        "   - Chat history",
        "   - Role/department context",
        "7. Assemble final prompt structure:",
        "   [ENHANCED_PREFIX]",
        "   CONTEXT: [filtered_docs]",
        "   QUESTION: [user_query]",
        "8. Send to LLM provider",
        "9. Process and store response"
    ]
    
    for step in flow_steps:
        print(f"  {step}")
    
    print("\nKey Components:")
    components = [
        ("Tone Guidance", "Analyzes user sentiment -> adjusts response style"),
        ("User Profile", "Name, role, department, skills, experience"),
        ("Chat History", "Last 5-10 conversation turns with timestamps"),
        ("RBAC Filtering", "Level-based access + role overrides"),
        ("Token Management", "Budget allocation and context window limits")
    ]
    
    for component, description in components:
        print(f"  - {component}: {description}")

if __name__ == "__main__":
    print("RAG Prompt Analysis Tool")
    print("=" * 40)
    
    show_prompt_flow()
    test_prompt_construction()
    test_with_llm()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nKey areas to check:")
    print("- Check logs for LLM_FINAL_PROMPT entries")
    print("- Monitor token usage in prompt construction")
    print("- Verify RBAC filtering effectiveness")
    print("- Test personalization with different user profiles")