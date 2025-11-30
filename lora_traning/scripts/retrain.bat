@echo off
REM Quick retraining script for Windows
REM Usage: retrain.bat

echo 🚀 Starting model training...

REM Install/update dependencies


REM Run training
echo Starting training process...
python scripts\train_model.py

REM Convert to GGUF if needed
if not exist "models\distilgpt2-company-tuned.gguf" (
    echo Converting to GGUF format...
    git clone https://github.com/ggerganov/llama.cpp.git temp_llama_cpp
    python temp_llama_cpp\convert_hf_to_gguf.py models\distilgpt2-company-tuned --outfile models\distilgpt2-company-tuned.gguf --outtype q8_0
    rmdir /s /q temp_llama_cpp
)

echo ✅ Training complete!
echo 📁 HuggingFace model: models\distilgpt2-company-tuned\
echo 📁 GGUF model: models\distilgpt2-company-tuned.gguf

pause