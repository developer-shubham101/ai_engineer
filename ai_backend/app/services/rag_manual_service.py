import os
import chromadb
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .google_models import google_llm
from .llm_service import TextRequest

load_dotenv()

# --- Global variables for our manual RAG setup ---
rag_collection = None
embedding_function = None

# --- CONFIGURATION CONSTANTS for Persistence (ADJUSTED FOR LOCAL DEV) ---
# We will create a local directory 'chroma_storage' in the root of the project
PERSIST_DIRECTORY = "./chroma_storage"  # <-- ADJUSTED PATH for local setup
COLLECTION_NAME = "apollo_mission"


def initialize_manual_rag():
    """
    Initializes a RAG pipeline manually using a persistent ChromaDB store.
    """
    global rag_collection, embedding_function

    if rag_collection is not None:
        print("Manual RAG already initialized.")
        return

    try:
        print(f"Initializing Manual RAG using persistent storage at {os.path.abspath(PERSIST_DIRECTORY)}...")

        # 1. Create embeddings function
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

        # 2. Initialize PERSISTENT ChromaDB client
        # This will create the directory if it doesn't exist and load the store.
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)  # <-- KEY CHANGE 1 (Still PersistentClient)

        # 3. Get or Create the Collection
        rag_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            # For the manual client, it's safer to ensure the embedding function is set explicitly here
            # using an adapter, or we rely on the query step to embed the text.
            # Sticking to the core client for simplicity here.
        )

        # 4. Conditional Load / Persistence Check
        current_count = rag_collection.count()
        if current_count == 0:
            print(f"INFO: Collection '{COLLECTION_NAME}' is empty. Initializing knowledge base from mission.txt...")

            # Load and Split the document
            # NOTE: Assuming mission.txt is at ./data/mission.txt relative to the project root
            # You might need to adjust the path if running from a different directory (e.g., from `main.py`).
            try:
                with open("./data/mission.txt", "r", encoding="utf-8") as f:
                    document_text = f.read()
            except FileNotFoundError:
                # Attempt to adjust path for common execution contexts
                with open("../data/mission.txt", "r", encoding="utf-8") as f:
                    document_text = f.read()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(document_text)

            # Generate embeddings and add them to the collection
            rag_collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))],
            )
            print(f"INFO: {len(chunks)} documents added to the persistent knowledge base.")

        else:
            print(f"INFO: Persistent knowledge base loaded with {current_count} documents. Skipping re-indexing.")


    except Exception as e:
        print(f"Error initializing Manual RAG: {e}")
        rag_collection = None


def query_manual_rag(request: TextRequest) -> str:
    # ... (No change to query logic) ...
    if not rag_collection or not google_llm:
        raise ConnectionError("Manual RAG is not initialized. Please check server logs.")

    try:
        # 1. Retrieve relevant documents MANUALLY
        retrieved_docs = rag_collection.query(
            query_texts=[request.text],
            n_results=3,
        )

        context_text = "\n\n---\n\n".join(retrieved_docs['documents'][0])

        prompt_template = f"""
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context_text}

        Question: {request.text}

        Helpful Answer:
        """

        response = google_llm.invoke(prompt_template)
        return response

    except Exception as e:
        raise ConnectionError(f"Failed to query manual RAG: {e}")


def add_document_to_rag(document_text: str, source_name: str = "api_upload") -> int:
    """
    Splits, embeds, and adds a new document's text to the existing ChromaDB collection.

    Returns:
        The number of chunks added.
    """
    global rag_collection, embedding_function

    if not rag_collection:
        raise ConnectionError("Manual RAG is not initialized.")

    if not document_text.strip():
        return 0

    print(f"INFO: Adding new document '{source_name}' to the knowledge base...")

    # 1. Split the document into chunks (Re-use the same splitter logic)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)

    # 2. Prepare metadata and IDs
    current_count = rag_collection.count()
    new_ids = [f"chunk_{current_count + i}" for i in range(len(chunks))]
    # Add simple metadata to track the source of the chunk
    metadata_list = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    # 3. Add to the collection
    rag_collection.add(
        documents=chunks,
        ids=new_ids,
        metadatas=metadata_list,
    )

    print(f"INFO: Successfully added {len(chunks)} chunks from '{source_name}'.")
    return len(chunks)