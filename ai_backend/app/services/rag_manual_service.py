# app/services/rag_manual_service.py

import os
import chromadb  # We will use the chromadb client directly
from dotenv import load_dotenv

# We still need some core LangChain components that are not in 'community'
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import the LLM we've already configured
from .google_models import google_llm
from .llm_service import TextRequest  # Re-use our Pydantic model

load_dotenv()

# --- Global variables for our manual RAG setup ---
# We will hold onto the ChromaDB collection object
rag_collection = None
# We still need the embedding function
embedding_function = None


def initialize_manual_rag():
    """
    Initializes a RAG pipeline manually, replacing langchain-community components.
    """
    global rag_collection, embedding_function

    if rag_collection is not None:
        print("Manual RAG already initialized.")
        return

    try:
        print("Initializing Manual RAG...")
        # 1. Load the document MANUALLY (Replaces TextLoader)
        with open("./data/mission.txt", "r", encoding="utf-8") as f:
            document_text = f.read()

        # 2. Split the document into chunks (We still use LangChain's splitter as it's a good utility)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document_text)

        # 3. Create embeddings function (This is still from a core LangChain package)
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

        # 4. Initialize ChromaDB client and create a collection MANUALLY (Replaces Chroma from_documents)
        chroma_client = chromadb.Client()  # Use the in-memory client
        rag_collection = chroma_client.create_collection(name="apollo_mission")

        # 5. Generate embeddings and add them to the collection
        rag_collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]  # Chroma needs unique IDs for each chunk
        )

        print("Manual RAG initialized successfully.")

    except Exception as e:
        print(f"Error initializing Manual RAG: {e}")
        rag_collection = None


def query_manual_rag(request: TextRequest) -> str:
    """Queries the manually built RAG pipeline."""
    if not rag_collection or not google_llm:
        raise ConnectionError("Manual RAG is not initialized. Please check server logs.")

    try:
        # 1. Retrieve relevant documents MANUALLY (Replaces retriever.get_relevant_documents)
        # We query the collection to get the most similar document chunks.
        retrieved_docs = rag_collection.query(
            query_texts=[request.text],
            n_results=3  # Ask for the top 3 most relevant chunks
        )

        # Extract the actual text from the retrieved documents
        context_text = "\n\n---\n\n".join(retrieved_docs['documents'][0])

        # 2. Build the prompt MANUALLY
        prompt_template = f"""
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context_text}

        Question: {request.text}

        Helpful Answer:
        """

        # 3. Call the LLM directly with the augmented prompt
        response = google_llm.predict(prompt_template)
        return response

    except Exception as e:
        raise ConnectionError(f"Failed to query manual RAG: {e}")