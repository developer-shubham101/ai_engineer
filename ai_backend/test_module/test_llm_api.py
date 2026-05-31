import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:8000/api/rag/llamaserver/query"

QUESTIONS = [
    "What is Trishul Dynamics Ltd.?",
    "Who founded Trishul Dynamics?",
    "Where is the company headquarters located?",
    "What subsidiaries does Trishul Dynamics own?",
    "What is the probation period for new employees?",
    "What was FY24 revenue?",
    "Show all subsidiaries and their headquarters.",
    "What is the salary of the CEO?",
    "What database does the HR portal use?",
    "Explain all AI governance policies."
]

payload_template = {
    "top_k": 3,
    "use_llm": True,
    "use_documents": True,
    "use_tools": False,
    "use_conversation_history": False,
    "temperature": 0.1,
    "max_tokens": 512,
    "conversation_id": "conv_2882976c3a3b484d84077391ab0ba763",
    "prompt_template": "enterprise_assistant"
}

results = []

for idx, question in enumerate(QUESTIONS, start=1):
    payload = payload_template.copy()
    payload["question"] = question

    try:
        start = time.time()

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
        )

        elapsed = round(time.time() - start, 2)

        data = response.json()

        results.append({
            "id": idx,
            "question": question,
            "answer": data.get("answer"),
            "latency_seconds": elapsed,
            "retrieved_docs": len(data.get("retrieved", [])),
            "raw_response": data
        })

        print(f"[PASS] {question}")

    except Exception as e:
        results.append({
            "id": idx,
            "question": question,
            "error": str(e)
        })

        print(f"[FAIL] {question}")

Path("rag_test_logs").mkdir(exist_ok=True)

with open(
    "rag_test_logs/results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(
    f"\nCompleted {len(QUESTIONS)} tests."
)
print(
    "Results saved to rag_test_logs/results.json"
)