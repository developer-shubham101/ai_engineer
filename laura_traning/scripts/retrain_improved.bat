@echo off
REM Complete training and conversion workflow for Windows
REM Usage: retrain_improved.bat

echo ========================================
echo COMPANY MODEL TRAINING WORKFLOW
echo ========================================

REM Step 1: Convert documents to text (optional)
echo.
echo Step 1a: Converting documents to text format...
python scripts\doc_parser.py

REM Step 1b: Prepare training data
echo.
echo Step 1b: Preparing training data from company documents...
python scripts\doc_parser.py
if errorlevel 1 (
    echo Error: Training data preparation failed!
    pause
    exit /b 1
)

REM Step 2: Run training
echo.
echo Step 3: Starting model training...
python scripts\train_model.py
if errorlevel 1 (
    echo Error: Model training failed!
    pause
    exit /b 1
)

REM Step 4: Convert to GGUF format
echo.
echo Step 4: Converting to GGUF format...
if exist "models\distilgpt2-company-tuned" (
    python scripts\convert_to_gguf_improved.py models\distilgpt2-company-tuned
    if errorlevel 1 (
        echo Warning: GGUF conversion failed, trying fallback method...
        python scripts\simple_gguf_convert_fixed.py models\distilgpt2-company-tuned
        if errorlevel 1 (
            echo Note: GGUF conversion failed, but HuggingFace model is available
        )
    )
) else (
    echo Error: Trained model directory not found!
)

REM Step 5: Summary
echo.
echo ========================================
echo TRAINING WORKFLOW COMPLETED!
echo ========================================
echo.
echo Available files:
if exist "models\distilgpt2-company-tuned" (
    echo    HuggingFace model: models\distilgpt2-company-tuned\
)
if exist "models\distilgpt2-company-tuned.gguf" (
    echo    GGUF model: models\distilgpt2-company-tuned.gguf
)
if exist "training_data.jsonl" (
    echo    Training data: training_data.jsonl
)

echo.
echo To test your model, run:
echo    python -m uvicorn app.main:app --reload
echo    Then use the /api/query endpoint
echo.

pause