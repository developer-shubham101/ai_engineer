# chroma_demo.py
import os
from embeddings import embed_texts
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

def run_demo():
    # persistent directory inside project
    persist_dir = os.path.join(os.getcwd(), ".chromadb")
    # Create in-memory ephemeral client (no files, easiest for testing)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # create or get collection
    col_name = "demo_collection"
    if col_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(col_name)
    else:
        collection = client.create_collection(col_name)

    # sample docs
    docs = [
        "Python is a programming language that emphasizes readability.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "ChromaDB provides a developer-friendly vector database.",
        "LanceDB is a file-backed vector database with an easy Python API.",
        "SQLite can be used as a tiny persistence layer for embeddings."
    ]
    ids = [f"doc{i}" for i in range(len(docs))]
    embs = embed_texts(docs).tolist()

    # add (upsert-like)
    # collection.add(documents=docs, embeddings=embs, ids=ids,
    #                metadatas=[{"source":"demo"}]*len(docs))

    # query
    query_text = "What is Python?"
    q_emb = embed_texts([query_text])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3, where=None)
    print("Query:", query_text)
    print("Results (documents):")
    for i, doc in enumerate(results["documents"][0]):
        print(f" - Rank {i+1}: {doc}")

if __name__ == "__main__":
    run_demo()
