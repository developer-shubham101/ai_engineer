from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_basic():
    res = client.post("/analyze/", json={"text": "hello world"})
    assert res.status_code == 200
    data = res.json()
    assert data["word_count"] == 2
    assert data["char_count"] == 11