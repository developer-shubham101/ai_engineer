#!/usr/bin/env python3
"""
Test script for trained company model.
Tests the model with various company policy questions.
"""

import requests
import json
import time
from typing import List, Dict

# Test server URL
BASE_URL = "http://192.168.1.2:8000"

# Test queries with expected content themes
TEST_QUERIES = [
    {
        "question": "What is the company policy on leave?",
        "expected_themes": ["privilege leave", "21 days", "casual leave", "sick leave", "wellness days"],
        "category": "HR Policy"
    },
    {
        "question": "Tell me about the attendance policy.",
        "expected_themes": ["core hours", "11:00 AM", "3:00 PM", "40 hours", "remote work"],
        "category": "HR Policy"
    },
    {
        "question": "What are the rules for moonlighting?",
        "expected_themes": ["strictly prohibited", "written consent", "HR", "legal"],
        "category": "Code of Conduct"
    },
    {
        "question": "How does parental leave work at the company?",
        "expected_themes": ["26 weeks", "primary caregivers", "4 weeks", "secondary caregivers"],
        "category": "Benefits"
    },
    {
        "question": "What is the dress code policy?",
        "expected_themes": ["smart casual", "formal attire", "client meetings"],
        "category": "Code of Conduct"
    },
    {
        "question": "Tell me about the company's health insurance.",
        "expected_themes": ["SecureLife World", "employee", "spouse", "2 children", "global health coverage"],
        "category": "Benefits"
    },
    {
        "question": "What are the core working hours?",
        "expected_themes": ["11:00 AM", "3:00 PM", "local time", "flexible work schedule"],
        "category": "Work Schedule"
    },
    {
        "question": "How is salary disbursed?",
        "expected_themes": ["last working day", "month", "variable pay", "performance bonuses"],
        "category": "Compensation"
    },
    {
        "question": "What is the remote work policy?",
        "expected_themes": ["remote-first", "Saarthi Infotech", "Vajra Solutions", "3-days-in-office", "hybrid policy"],
        "category": "Work Arrangement"
    },
    {
        "question": "Tell me about wellness days.",
        "expected_themes": ["4 days", "unplug", "Aisha Sharma", "burnout", "once per quarter"],
        "category": "Benefits"
    }
]

def query_model(question: str, model_name: str = None) -> Dict:
    """Send a query to the model and return the response."""
    
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.config.model_config import ModelConfig
    
    if model_name is None:
        model_name = ModelConfig.DEFAULT_OUTPUT_NAME
    
    payload = {
        "question": question,
        "use_llm": True,
        "max_tokens": 300,
        "debug": True,
        "local_llm_model": model_name
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "status_code": response.status_code
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "status_code": None
        }

def evaluate_response(response: str, expected_themes: List[str]) -> Dict:
    """Evaluate if the response contains expected themes."""
    
    response_lower = response.lower()
    found_themes = []
    missing_themes = []
    
    for theme in expected_themes:
        if theme.lower() in response_lower:
            found_themes.append(theme)
        else:
            missing_themes.append(theme)
    
    score = len(found_themes) / len(expected_themes) if expected_themes else 0
    
    return {
        "score": score,
        "found_themes": found_themes,
        "missing_themes": missing_themes,
        "total_themes": len(expected_themes)
    }

def run_tests():
    """Run all test queries and evaluate responses."""
    
    print("TESTING TRAINED COMPANY MODEL")
    print("=" * 50)
    
    # Check if server is running
    # try:
    #     response = requests.get(f"{BASE_URL}/api/models/list", timeout=5)
    #     if response.status_code != 200:
    #         print("Server not responding. Please start the server with:")
    #         print("   python -m uvicorn app.main:app --reload")
    #         return
    # except requests.exceptions.RequestException:
    #     print("Cannot connect to server. Please start the server with:")
    #     print("   python -m uvicorn app.main:app --reload")
    #     return
    
    results = []
    total_score = 0
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\nTest {i}/{len(TEST_QUERIES)}: {test_case['category']}")
        print(f"Question: {test_case['question']}")
        
        # Send query
        result = query_model(question=test_case['question'])
        
        if not result['success']:
            print(f"Query failed: {result['error']}")
            results.append({
                "test_case": test_case,
                "success": False,
                "error": result['error']
            })
            continue
        
        # Get response
        answer = result['data']['answer']
        print(f"Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        # Evaluate response
        evaluation = evaluate_response(answer, test_case['expected_themes'])
        total_score += evaluation['score']
        
        print(f"Score: {evaluation['score']:.2f} ({len(evaluation['found_themes'])}/{evaluation['total_themes']} themes found)")
        
        if evaluation['found_themes']:
            print(f"Found themes: {', '.join(evaluation['found_themes'])}")
        
        if evaluation['missing_themes']:
            print(f"Missing themes: {', '.join(evaluation['missing_themes'])}")
        
        results.append({
            "test_case": test_case,
            "success": True,
            "response": answer,
            "evaluation": evaluation
        })
        
        time.sleep(1)  # Brief pause between requests
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    avg_score = 0
    if successful_tests:
        avg_score = total_score / len(successful_tests)
        print(f"Successful tests: {len(successful_tests)}/{len(TEST_QUERIES)}")
        print(f"Average score: {avg_score:.2f}")
        
        # Category breakdown
        category_scores = {}
        for result in successful_tests:
            category = result['test_case']['category']
            score = result['evaluation']['score']
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
        
        print("\nPerformance by category:")
        for category, scores in category_scores.items():
            avg_cat_score = sum(scores) / len(scores)
            print(f"   {category}: {avg_cat_score:.2f}")
    
    if failed_tests:
        print(f"\nFailed tests: {len(failed_tests)}")
        for result in failed_tests:
            print(f"   - {result['test_case']['question']}: {result['error']}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if avg_score < 0.3:
        print("   - Model needs more training or better data preparation")
        print("   - Consider increasing training epochs or data quality")
    elif avg_score < 0.6:
        print("   - Model shows some knowledge but needs improvement")
        print("   - Consider fine-tuning hyperparameters or adding more data")
    else:
        print("   - Model performs well on company policy questions")
        print("   - Ready for production use with monitoring")

def main():
    """Main test function."""
    run_tests()

if __name__ == "__main__":
    main()