from app.services.rag_local_service import initialize_local_rag, add_document_to_rag_local, query_local_rag

initialize_local_rag()  # will initialize Chroma; embedding model will load on demand
ids = add_document_to_rag_local("test", "This is a short test document about ducks and ponds.")
print("Indexed:", ids)
res = query_local_rag("What is this about?", n_results=2, use_llm=False)
print("Retrieved docs:", res["documents"])
