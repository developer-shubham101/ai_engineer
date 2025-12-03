#!/usr/bin/env python3
"""
Model evaluation script for trained company model.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

QUESTIONS = [
    "Passwords must be changed every 90 days"
    # "Tell me about the attendance policy.",
    # "What are the rules for moonlighting?",
    # "How does parental leave work at the company?",
    # "What is the dress code policy?",
    # "Tell me about the company's health insurance.",
    # "What are the core working hours?",
    # "How is salary disbursed?",
    # "What is the remote work policy?",
    # "Tell me about wellness days."
]

def ask_question(question):
    payload = {
        "question": question,
        "use_llm": True,
        "max_tokens": 300,
        "debug": True,
        "local_llm_model": "distilgpt2-company-tuned"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=330)
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Failed: {str(e)}"

def main():
    print("COMPANY MODEL EVALUATION")
    print("=" * 40)
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{i}. {question}")
        answer = ask_question(question)
        # print(f"Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(f"Answer: {answer}")
        time.sleep(1)

if __name__ == "__main__":
    main()