@echo off
REM LoRA Training Pipeline with JSONL Data
REM Usage: retrain_improved.bat

echo.
echo ============================================
echo      LoRA Training Pipeline (JSONL)
echo ============================================
echo.
echo Using data from: data\agniholdings_train.jsonl
echo.
echo Press Ctrl+C to cancel or
pause

REM Step 1: Install/Update dependencies
echo.
echo [Step 1/4] Installing dependencies...
@REM pip install -q transformers datasets torch accelerate peft

REM Step 2: Train model (uses JSONL data directly)
echo.
echo [Step 2/4] Training model on company data...
echo.
python scripts\train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed!
    pause
    exit /b 1
)

REM Step 3: Convert to GGUF format
echo.
echo [Step 3/4] Converting to GGUF format...
echo.
python scripts\convert_to_gguf_improved.py models\gpt2-company-tuned
if errorlevel 1 (
    echo WARNING: GGUF conversion failed, but HuggingFace model is available
)

REM Step 4: Show summary
echo.
echo [Step 4/4] Training complete!
echo.
echo ============================================
echo               SUMMARY
echo ============================================
echo.
echo Data source: data\agniholdings_train.jsonl
echo Training samples: 12
echo.
echo Available files:
if exist "models\gpt2-company-tuned\" echo   - HuggingFace model: models\gpt2-company-tuned\
if exist "models\gpt2-company-tuned.gguf" echo   - GGUF model: models\gpt2-company-tuned.gguf
if exist "models\gpt2-company-tuned.json" echo   - Training info: models\gpt2-company-tuned.json
echo.
echo ============================================
echo.
echo To test the model:
echo   1. Start server: python -m uvicorn app.main:app --reload
echo   2. Run tests:    python scripts\test_trained_model.py
echo.
pause