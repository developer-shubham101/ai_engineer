# lancedb_demo.py
import os
import numpy as np
import lancedb
from embeddings import embed_texts

def run_demo():
    # local relative folder for the db (works on Windows)
    path = os.path.join(os.getcwd(), "lance_db")
    uri = f"file:///{path.replace(os.sep, '/')}"  # safe file URI for lancedb
    db = lancedb.connect(uri)

    # simple table creation (Lance will create it on first add)
    texts = [
        "The dog chased the ball.",
        "Cats are independent animals.",
        "Birds can fly in the sky.",
        "FAISS and Chroma are vector DB examples.",
    ]
    embs = embed_texts(texts).astype(np.float32)

    # create table and add rows
    if "demo_table" in [t.name for t in db.list_tables()]:
        table = db.get_table("demo_table")
    else:
        # lightweight row-by-row add (small demo)
        table = db.create_table("demo_table", schema={"text": str, "embedding": f"vector<float32>[{embs.shape[1]}]"})
    # add batch
    rows = [{"text": texts[i], "embedding": embs[i]} for i in range(len(texts))]
    table.add(rows)

    # search (API: table.search)
    q_emb = embed_texts(["animals that fly"]).astype(np.float32)
    # note: exact API may vary across versions; this is a minimal example
    results = table.search(q_emb, k=3)  # returns objects or rows depending on version
    print("Search results (raw):", results)

if __name__ == "__main__":
    run_demo()
