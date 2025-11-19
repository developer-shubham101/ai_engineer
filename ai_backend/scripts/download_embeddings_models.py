from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = Path("embeddings_models/all-MiniLM-L6-v2/embeddings_models/all-MiniLM-L6-v2")

print("Downloading MiniLM embedding model...")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)
model.save(str(LOCAL_DIR))

print("Done! Model saved to:", LOCAL_DIR.resolve())
