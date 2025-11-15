# sqlite_demo.py
import sqlite3
import numpy as np
from embeddings import embed_texts
import os

DB_PATH = "embs.sqlite"

def to_blob(arr: np.ndarray):
    return arr.tobytes()

def from_blob(b: bytes, dim: int):
    return np.frombuffer(b, dtype=np.float32).reshape(dim)

def create_and_populate():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT, emb BLOB)")
    conn.commit()

    texts = [
        "Bananas are yellow and rich in potassium.",
        "Oranges are citrus fruits with vitamin C.",
        "Apples are often red or green and crisp."
    ]
    embs = embed_texts(texts).astype(np.float32)
    for i, t in enumerate(texts):
        cur.execute("INSERT OR REPLACE INTO docs (id, text, emb) VALUES (?, ?, ?)", (str(i), t, to_blob(embs[i])))
    conn.commit()
    conn.close()

def query(q_text, top_k=3):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q_emb = embed_texts([q_text]).astype(np.float32)[0]
    rows = cur.execute("SELECT id, text, emb FROM docs").fetchall()
    sims = []
    for id_, text, b in rows:
        emb = from_blob(b, q_emb.shape[0])
        cos = float((q_emb @ emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
        sims.append((id_, text, cos))
    sims.sort(key=lambda x: x[2], reverse=True)
    conn.close()
    return sims[:top_k]

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        create_and_populate()
    top = query("Which fruit has vitamin C?")
    print("Top matches:")
    for r in top:
        print(r)
