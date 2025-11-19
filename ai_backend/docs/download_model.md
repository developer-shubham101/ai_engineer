### ✅ Good model options

1. Phi‑3‑mini‑4K‑Instruct‑GGUF (by Microsoft)

   * Repo: `microsoft/Phi-3-mini-4k-instruct-gguf`. ([Hugging Face][1])
   * Size: ~ 3.8B parameters. GGUF quantised files (for example Q4_K, Q3_K) 2-5 GB/less memory usage. ([Hugging Face][1])
   * License: MIT. ([Hugging Face][1])
   * Good fit for CPU + local inference.
   * **Download command:**

     ```bash
     huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models/
     ```

2. Llama 2 7B ‑ GGUF version (by TheBloke)

   * Repo: `TheBloke/Llama-2-7B-GGUF`. ([Hugging Face][2])
   * Contains multiple quantisation variants, e.g., Q4_K_M (~4 GB) etc. Good for CPU. ([Hugging Face][2])
   * License and usage appear open.
   * **Download command:**

     ```bash
     huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir models/
     ```

3. Mistral 7B Instruct v0.2 GGUF (by Mistral AI / TheBloke)

   * Repo: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`. ([Hugging Face][3])
   * Contains very CPU-friendly quantised GGUF variants (2-3 GB versions) so suitable for modest hardware.
   * **Download command:**

     ```bash
     huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q3_K_M.gguf --local-dir models/
     ```

---

### ℹ️ Notes before download

* Make sure you are logged into Hugging Face CLI if the repo requires authentication (some open models may still require Accepting license/terms).
* Use a path like `models/` in your project root so your local code can find `LLM_MODEL_FILENAME = <filename>.gguf`.
* Since you are using **CPU only**, pick the smallest quantised variant (e.g., Q3_K or Q4_K) to reduce memory load.
* After download, your model directory might contain multiple files or variants – pick the one you reference in config.

---

### 🛠 Example steps for you (Windows/PyCharm venv)

```powershell
# Activate your venv
.\.venv\Scripts\activate

# (Optional) install hug-face hub CLI if not installed
pip install huggingface-hub

# Download model, e.g. Mistral 7B
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q3_K_M.gguf --local-dir models/

# After download, verify that file exists:
dir .\models\mistral-7b-instruct-v0.2.Q3_K_M.gguf

# Set environment variable (or update config) in your project:
set LLM_MODEL_FILENAME=mistral-7b-instruct-v0.2.Q3_K_M.gguf

# Start the server
uvicorn app.main:app --reload --port 8000
```