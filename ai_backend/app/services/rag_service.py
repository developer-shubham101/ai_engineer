# app/services/rag_service.py

import os
from dotenv import load_dotenv

# LangChain components for RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For creating embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Import the LLM we've already configured
from .google_models import google_llm

load_dotenv()

# Global variable to hold the RAG chain
rag_chain = None


def initialize_rag_chain():
    """
    Initializes the RAG chain by loading a document, splitting it,
    creating embeddings, and storing them in a vector database.
    """
    global rag_chain

    if rag_chain is not None:
        print("RAG chain already initialized.")
        return

    try:
        print("Initializing RAG chain...")
        # 1. Load the document
        # In a real app, this path could come from a config file or user input
        loader = TextLoader("./data/mission.txt")
        documents = loader.load()

        # 2. Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings for the text chunks
        # We use a Google Generative AI model for this.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

        # 4. Store the embeddings in a Chroma vector database
        # This is an in-memory vector store, perfect for getting started.
        # It will disappear when the application shuts down.
        vector_store = Chroma.from_documents(texts, embeddings)

        # 5. Create the RetrievalQA chain
        # This chain combines a retriever (our vector store) and an LLM.
        rag_chain = RetrievalQA.from_chain_type(
            llm=google_llm,
            chain_type="stuff",  # "stuff" means it stuffs all retrieved text into the prompt
            retriever=vector_store.as_retriever()
        )
        print("RAG chain initialized successfully.")

    except Exception as e:
        print(f"Error initializing RAG chain: {e}")
        rag_chain = None


# --- Service Function for the RAG endpoint ---

from .llm_service import TextRequest  # Re-use our Pydantic model


def query_document(request: TextRequest) -> str:
    """Queries the initialized RAG chain."""
    if not rag_chain:
        raise ConnectionError("RAG chain is not initialized. Please check server logs.")

    try:
        # The chain takes the query, finds relevant docs, formats the prompt, and gets the answer.
        result = rag_chain.run(request.text)
        return result
    except Exception as e:
        raise ConnectionError(f"Failed to query document: {e}")