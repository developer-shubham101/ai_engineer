```
Failed to initialize local LlamaCpp instance:
Could not import llama-cpp-python library.
Please install the llama-cpp-python library to use this embedding model:
pip install llama-cpp-python
```

This means **llama-cpp-python did NOT install correctly** on your Windows machine.

This is **VERY common** on Windows, because llama-cpp-python requires native C++ compilation unless a prebuilt wheel exists for your Python version.

Let me walk you through fixing it **step-by-step**.

---

# ✅ **1. Check your Python version**

You said earlier:

✔ Python **3.10**
✔ Windows
✔ Using PyCharm venv

Good — that means we *can* install llama-cpp-python, but we need Build Tools.

---

# ✅ **2. Install Microsoft C++ Build Tools (required!)**

### You MUST install this on Windows:

👉 Download:
[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### During installation:

✔ Select **"Desktop development with C++"**
✔ Make sure these are checked:

* MSVC v14.x C++ compiler toolset
* Windows 10/11 SDK
* CMake
* Build Tools for C++
* Ninja (optional, helps with builds)

After installation: **restart your computer**.

---

# ✅ **3. Reinstall llama-cpp-python inside your venv**

Activate your venv:

```powershell
.\.venv\Scripts\activate
```

Then:

```powershell
pip install --upgrade pip wheel setuptools
pip install llama-cpp-python --verbose
```

If this succeeds → GREAT 🎉

If you get errors like "cl.exe not found" → Build Tools are not installed correctly.

---

# ❗ 4. If the build still fails, use a **precompiled wheel**

Good news: there are prebuilt wheels for Windows if you match Python version and CPU architecture.

You can install a CPU-only wheel using:

```powershell
pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python/whl/cpu/
```

This installs a **precompiled Windows CPU wheel**.

This works most of the time.

---

# ⭐ 5. Verify installation

```powershell
python -c "from llama_cpp import Llama; print('OK')"
```

If you see **OK**, you’re ready 🎉

---

# ⭐ 6. Make sure your model filename is correct

Your error happened *before* even loading the model, but check anyway:

Set the environment variable:

```powershell
set LLM_MODEL_FILENAME=mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

or in PowerShell:

```powershell
$env:LLM_MODEL_FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

Restart FastAPI:

```powershell
uvicorn app.main:app --reload --port 8000
```

---

# 7. If you STILL get errors — tell me which one:

I need to see:

* The exact console error
* Your Python version (`python --version`)
* The output of:

```powershell
pip show llama-cpp-python
```