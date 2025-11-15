# rag_demo.py
from embeddings import embed_texts
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

def get_chroma_collection():
    persist_dir = os.path.join(os.getcwd(), ".chromadb")
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    col_name = "demo_collection"
    if col_name in [c.name for c in client.list_collections()]:
        return client.get_collection(col_name)
    else:
        raise RuntimeError("Collection not found. Run chroma_demo.py to populate the collection first.")

def build_prompt(question, retrieved_docs):
    prompt = "You are a helpful assistant. Use the provided documents to answer the question.\n\n"
    for i, doc in enumerate(retrieved_docs):
        prompt += f"Document {i+1}:\n{doc}\n\n---\n\n"
    prompt += f"Question: {question}\nAnswer concisely:"
    return prompt

def run_rag_demo():
    collection = get_chroma_collection()
    user_query = "Tell me briefly about FAISS."
    q_emb = embed_texts([user_query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3)
    docs = res["documents"][0]
    prompt = build_prompt(user_query, docs)
    print("=== PROMPT SENT TO LLM ===")
    print(prompt)
    # At this point you would call an LLM API (OpenAI, local LLM) with 'prompt'.
    # For learning locally, you can use a small local LLM or simply inspect the prompt.

if __name__ == "__main__":
    run_rag_demo()
