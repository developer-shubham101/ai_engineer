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
REM Get model name from config and convert
for /f "tokens=*" %%i in ('python -c "import sys; sys.path.insert(0, '.'); from app.config.model_config import ModelConfig; print(ModelConfig.DEFAULT_OUTPUT_NAME)"') do set MODEL_NAME=%%i

if exist "models\%MODEL_NAME%" (
    python scripts\convert_to_gguf_improved.py models\%MODEL_NAME%
    if errorlevel 1 (
        echo Warning: GGUF conversion failed, trying fallback method...
        python scripts\simple_gguf_convert_fixed.py models\%MODEL_NAME%
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
if exist "models\%MODEL_NAME%" (
    echo    HuggingFace model: models\%MODEL_NAME%\
)
if exist "models\%MODEL_NAME%.gguf" (
    echo    GGUF model: models\%MODEL_NAME%.gguf
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