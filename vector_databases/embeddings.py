# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Choose a compact CPU-friendly model
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-d, fast on CPU

def get_model():
    """Load and return the SentenceTransformer model (singleton-style)."""
    global _model
    try:
        _model
    except NameError:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts, batch_size=32):
    """Return numpy array float32 embeddings for a list of texts."""
    model = get_model()
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embs, dtype=np.float32)
