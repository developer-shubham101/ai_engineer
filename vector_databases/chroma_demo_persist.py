# chroma_demo_persist.py
import os
import time
from embeddings import embed_texts
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

def get_persistent_client(persist_dir: str):
    """
    Create or open a persistent Chroma client using the new PersistentClient API.
    The directory will be created if missing.
    """
    # ensure directory exists (Chroma will create files inside)
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(),            # default settings are fine for learning
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return client

def get_or_create_collection(client, col_name: str):
    """Return an existing collection or create a new one if missing."""
    try:
        # Preferred: try to get collection first
        return client.get_collection(col_name)
    except Exception:
        # If collection doesn't exist, create it
        return client.create_collection(col_name)

def populate_demo_collection(collection):
    """Add example docs if the collection is empty (idempotent-ish)."""
    # Check if collection is non-empty
    metadata = collection.count() if hasattr(collection, "count") else None
    # Some versions may not have .count() on collection; fall back to safe try:
    try:
        existing = collection.count()
    except Exception:
        # best-effort: try listing documents via query of empty embedding
        existing = 0
    if existing and existing > 0:
        print(f"Collection already has {existing} items — skipping population.")
        return

    docs = [
        "Python is a programming language that emphasizes readability and developer productivity.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors (from Facebook/Meta).",
        "ChromaDB provides an easy-to-use vector store for prototypes and RAG workflows.",
        "LanceDB is an efficient file-backed vector database.",
        "SQLite can be used as a very small persistence backend for embeddings via BLOBs."
    ]
    ids = [f"doc{i}" for i in range(len(docs))]
    embs = embed_texts(docs).tolist()

    print(f"Adding {len(docs)} documents to collection...")
    collection.add(documents=docs, embeddings=embs, ids=ids,
                   metadatas=[{"source":"demo"}]*len(docs))
    print("Added documents.")

def verify_persistence(persist_dir: str):
    """Print a simple listing of the directory so you can see Chroma files."""
    print("\nFiles under persist directory:")
    for root, dirs, files in os.walk(persist_dir):
        level = root.replace(persist_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  - {f}")

def run_demo():
    persist_dir = os.path.join(os.getcwd(), ".chromadb")  # will be created
    client = get_persistent_client(persist_dir)

    col_name = "demo_collection"
    collection = get_or_create_collection(client, col_name)

    # Populate only if empty
    populate_demo_collection(collection)

    # Persist to disk (ensures files are flushed)
    print("Persisting client to disk...")
    try:
        client.persist()
    except Exception as e:
        # Some chroma versions auto-persist or don't expose persist on PersistentClient in the same way
        print("client.persist() raised:", e)
        print("Proceeding — data may still have been written by Chroma automatically.")

    # Small delay to ensure FS sync (not strictly necessary)
    time.sleep(0.5)

    # Verify files exist
    verify_persistence(persist_dir)

    # Run a test query
    q_text = "Tell me about FAISS"
    q_emb = embed_texts([q_text])[0].tolist()
    print("\nRunning a query:", q_text)
    res = collection.query(query_embeddings=[q_emb], n_results=3)
    print("Documents returned:")
    for i, doc in enumerate(res.get("documents", [[]])[0]):
        print(f" {i+1}. {doc}")

if __name__ == "__main__":
    run_demo()
