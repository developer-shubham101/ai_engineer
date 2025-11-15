# faiss_demo.py
import numpy as np
import faiss
from embeddings import embed_texts

def faiss_flat_search(embs, query_emb, k=3):
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)  # exact L2
    index.add(embs)
    D, I = index.search(query_emb, k)
    return D, I

def faiss_hnsw_search(embs, query_emb, k=3):
    d = embs.shape[1]
    M = 32  # HNSW parameter (higher = more accurate, more memory)
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 40
    index.add(embs)
    index.hnsw.efSearch = 16
    D, I = index.search(query_emb, k)
    return D, I

def run_demo():
    docs = [
        "Apple makes consumer electronics and software.",
        "Microsoft builds developer tools and cloud services.",
        "NVIDIA builds GPUs for compute and AI workloads.",
        "FAISS is for fast similarity search.",
        "SentenceTransformers convert text to dense vectors."
    ]
    embs = embed_texts(docs)  # numpy float32
    query = ["What is FAISS?"]
    q_emb = embed_texts(query)

    Df, If = faiss_flat_search(embs, q_emb, k=3)
    Dh, Ih = faiss_hnsw_search(embs, q_emb, k=3)

    print("Exact (Flat) nearest indices:", If[0], "distances:", Df[0])
    print("HNSW nearest indices:", Ih[0], "distances:", Dh[0])

    print("\nMatched documents (Flat):")
    for idx in If[0]:
        print(" -", docs[idx])

if __name__ == "__main__":
    run_demo()
