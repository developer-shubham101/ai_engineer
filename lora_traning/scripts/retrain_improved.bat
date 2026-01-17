@echo off
REM LoRA Training Pipeline (Llama 3.2 1B + Alpaca)
REM Usage: retrain_improved.bat

echo.
echo ============================================
echo      LoRA Training Pipeline (Llama 3.2)
echo ============================================
echo.
echo Workflow:
echo 1. Download Base Model (Llama 3.2 1B Instruct)
echo 2. Train Model (uses yahma/alpaca-cleaned by default)
echo 3. Convert to GGUF format
echo.
echo Press Ctrl+C to cancel or
pause

REM Step 0: Download Base Model
echo.
echo [Step 1/3] Downloading/Verifying Base Model...
python scripts\download_model.py
if errorlevel 1 (
    echo ERROR: Model download failed!
    pause
    exit /b 1
)

REM Step 2: Train model
echo.
echo [Step 2/3] Training model...
echo.
python scripts\train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed!
    pause
    exit /b 1
)

REM Step 3: Convert to GGUF format
echo.
echo [Step 3/3] Converting to GGUF format...
echo.
REM Note: train_model.py outputs new model name, we try to use default if possible
REM But simplest way is to point to the directory config uses.
REM Since we don't know exact output name here without parsing config, 
REM let's assume default "llama-3.2-1b-instruct-company-tuned"
python scripts\convert_to_gguf_improved.py models\llama-3.2-1b-instruct-company-tuned
if errorlevel 1 (
    echo WARNING: GGUF conversion failed, but HuggingFace model might be available in models/
)

echo.
echo ============================================
echo               SUMMARY
echo ============================================
echo.
echo Base Model: meta-llama/Llama-3.2-1B-Instruct
echo Dataset: yahma/alpaca-cleaned (Twice-cached: HF cache + Model cache)
echo.
echo To test the model:
echo   1. Start server: python -m uvicorn app.main:app --reload
echo   2. Run tests:    python scripts\test_trained_model.py
echo.
pause