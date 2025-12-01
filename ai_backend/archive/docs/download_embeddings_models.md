# ✅ **Method 1 — Easiest (Python auto-download into your project folder)**

This downloads the model directly into:

```
your_project/embeddings_models/all-MiniLM-L6-v2/
```

📌 **Create a file**: `download_minilm.py`
Paste this:

```python
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = Path("../embeddings_models/all-MiniLM-L6-v2")

print("Downloading MiniLM embedding model...")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)
model.save(str(LOCAL_DIR))

print("Done! Model saved to:", LOCAL_DIR.resolve())
```

### Run it:

```powershell
.\.venv\Scripts\activate
python download_minilm.py
```

✔ Downloads
✔ Saves locally
✔ No need for internet again
✔ Your RAG system will load from this folder automatically

---

# ✅ **Method 2 — HuggingFace CLI download (recommended if HF installed)**

Install CLI:

```powershell
pip install huggingface-hub
```

Log in (optional):

```powershell
huggingface-cli login
```

Download:

```powershell
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir embeddings_models/all-MiniLM-L6-v2
```

This creates:

```
embeddings_models/all-MiniLM-L6-v2/
    ├── config.json
    ├── pytorch_model.bin
    ├── sentence_bert_config.json
    ├── tokenizer.json
    └── ...
```

Your code will detect this automatically via:

```python
_get_local_embedding_model_path()
```

---

# ✅ **Method 3 — Manual Download (if you prefer browser)**

Go to:

🔗 [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main)

Click **Download all files** (or each file):

* `config.json`
* `pytorch_model.bin`
* `sentence_bert_config.json`
* `tokenizer.json`
* `vocab.txt` (if present)

Create folder:

```
embeddings_models/all-MiniLM-L6-v2/
```

Put all files inside it.

Your folder will look like:

```
embeddings_models/
│
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── sentence_bert_config.json
    ├── vocab.txt     (maybe)
```

Now your project loads it instantly — no huggingface needed at runtime.

---

# 🎯 **Verify it's working**

Run this:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("embeddings_models/all-MiniLM-L6-v2")
print("Loaded model successfully!")
```

If you see:

```
Loaded model successfully!
```

→ Perfect, embeddings will now stay offline.

---

# 🎉 **What your RAG pipeline does with this local model**

Your `rag_local_service.py` checks:

```python
local_path = BASE_DIR.parent / "embeddings_models" / "all-MiniLM-L6-v2"
if local_path.exists():
    load from disk
else:
    download online
```

So your local folder ensures **offline embeddings**, fast loading, no surprises.