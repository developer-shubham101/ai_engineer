"""Download all embedding models defined in settings.EMBEDDING_MODELS."""
from __future__ import annotations

from pathlib import Path

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    # "e5-base-v2": "intfloat/e5-base-v2",
    # "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}

BASE_DIR = Path("embeddings_models")


def download_all() -> None:
    from sentence_transformers import SentenceTransformer

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for key, model_name in EMBEDDING_MODELS.items():
        local_dir = BASE_DIR / key
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[SKIP] {key} already exists at {local_dir.resolve()}")
            continue

        print(f"[DOWNLOAD] {key} ({model_name}) ...")
        try:
            model = SentenceTransformer(model_name)
            model.save(str(local_dir))
            print(f"[OK]   Saved to {local_dir.resolve()}")
        except Exception as e:
            print(f"[ERROR] {key}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    download_all()
