import requests
import json
import sys
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://192.168.1.2:8000"  # User provided URL
# BASE_URL = "http://localhost:5444" # Fallback/Local testing
USERNAME = "admin"
PASSWORD = "admin123"

def print_step(step: str):
    print(f"\n{'='*50}")
    print(f"STEP: {step}")
    print(f"{'='*50}")

def print_success(msg: str):
    print(f"✅ SUCCESS: {msg}")

def print_error(msg: str):
    print(f"❌ ERROR: {msg}")

def get_token() -> str:
    """Login and get JWT token"""
    print_step("Authenticating")
    url = f"{BASE_URL}/api/auth/token"
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        token = response.json().get("access_token")
        print_success("Authenticated successfully")
        return token
    except Exception as e:
        print_error(f"Authentication failed: {e}")
        if response.text:
            print(f"Response: {response.text}")
        sys.exit(1)

def check_documents(token: str) -> List[Dict]:
    """List all documents"""
    print_step("Checking Existing Documents")
    url = f"{BASE_URL}/api/rag/documents/list?latest_only=true"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        docs = response.json().get("documents", [])
        print(f"Found {len(docs)} documents.")
        for doc in docs:
            print(f" - {doc.get('source_name')} (v{doc.get('version')})")
        return docs
    else:
        print_error(f"Failed to list documents: {response.text}")
        return []

def seed_data(token: str):
    """Trigger seeding if needed"""
    print_step("Seeding Data")
    url = f"{BASE_URL}/api/rag/documents/seed?reseed=true" # Force reseed to ensure we have the data
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print_success(f"Seeding complete: {response.json().get('message')}")
    else:
        print_error(f"Seeding failed: {response.text}")

def test_rag_query(token: str):
    """Test RAG query for specific version changes"""
    print_step("Testing RAG Query (Leave Policy Changes)")
    
    # Question specifically targeting the evolution of the policy
    question = "What are the new changes in the leave policy? How many days do we get now compared to before?"
    
    url = f"{BASE_URL}/api/rag/local/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "top_k": 5,
        "use_llm": False,
        "debug": True
    }
    
    print(f"Question: {question}")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        answer = data.get("answer")
        print("\n🤖 RAG Answer:")
        print(f"{'-'*20}")
        print(answer)
        print(f"{'-'*20}")
        
        # Basic validation of the answer
        if "25 days" in answer or "20 days" in answer:
            print_success("Answer mentions updated leave counts (20 or 25 days).")
        else:
            print("⚠️ Warning: Answer might not contain expected specific details. Check manually.")
            
        # Check retrieved sources
        print("\nSources used:")
        for doc in data.get("retrieved", []):
            meta = doc.get("metadata", {})
            print(f" - {meta.get('source_name')} (v{meta.get('version')})")
            
    else:
        print_error(f"Query failed: {response.text}")

def test_crud_flow(token: str):
    """Test Add -> Update -> Archive flow"""
    print_step("Testing CRUD Flow (Add -> Update -> Archive)")
    
    # 1. Add Document
    print("1. Adding new document (v1.0)...")
    add_url = f"{BASE_URL}/api/rag/documents/add"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "source_name": "test_protocol.md",
        "text": "Test Protocol v1.0\n\nThis is a test document.",
        "metadata": {"department": "IT"}
    }
    res = requests.post(add_url, headers=headers, json=payload)
    if res.status_code != 200:
        print_error(f"Add failed: {res.text}")
        return
    
    # Extract document_id from the response message or by listing
    # The current API returns a message string with the ID, let's parse or fetch list
    # "Added document 'test_protocol.md' (v1.0, 1 chunks, document_id=...)"
    # A cleaner way is to fetch the list and find it
    time.sleep(1) # Give it a moment
    docs = check_documents(token)
    target_doc = next((d for d in docs if "test_protocol" in d.get("source_name", "")), None)
    
    if not target_doc:
        print_error("Could not find added document.")
        return
    
    doc_id = target_doc['document_id']
    print_success(f"Document added. ID: {doc_id}")
    
    # 2. Update Document
    print(f"\n2. Updating document {doc_id} (creating v2.0)...")
    update_url = f"{BASE_URL}/api/rag/documents/update"
    update_payload = {
        "document_id": doc_id,
        "text": "Test Protocol v2.0\n\nThis is the UPDATED test document.",
        "version_notes": "Updated content for testing",
        "status": "published"
    }
    res = requests.post(update_url, headers=headers, json=update_payload)
    if res.status_code == 200:
        print_success("Document updated successfully (v2.0 created).")
    else:
        print_error(f"Update failed: {res.text}")
        
    # 3. Archive Version
    print(f"\n3. Archiving v1.0 of {doc_id}...")
    archive_url = f"{BASE_URL}/api/rag/documents/{doc_id}/archive"
    params = {"version": "1.0"}
    res = requests.post(archive_url, headers=headers, params=params)
    if res.status_code == 200:
        print_success("Version 1.0 archived successfully.")
    else:
        print_error(f"Archive failed: {res.text}")

def main():
    print("🚀 Starting App Verification Script")
    print(f"Target: {BASE_URL}")
    
    # 1. Authenticate
    token = get_token()
    
    # 2. Check & Seed
    docs = check_documents(token)
    
    # Check if we have the expected data
    has_policy = any("leave_policy" in d.get("source_name", "") for d in docs)
    
    if not has_policy:
        print("Target data (leave_policy) not found. Forcing seed...")
        seed_data(token)
        check_documents(token)
    else:
        print("Target data found. Skipping seed.")

    # 3. Test RAG capabilities
    test_rag_query(token)
    
    # 4. Test CRUD capabilities
    test_crud_flow(token)
    
    print("\n✨ Verification Complete!")

if __name__ == "__main__":
    main()
