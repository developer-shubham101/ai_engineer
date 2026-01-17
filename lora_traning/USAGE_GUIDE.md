# 🚀 Company Model Training & Testing Guide

## Quick Start

### 1. Complete Training Workflow (Recommended)
```bash
# Run the complete workflow - data prep, training, and conversion
scripts\retrain_improved.bat
```

This will:
- ✅ Prepare training data from `data/` folder (.md and .txt files)
- ✅ Install dependencies
- ✅ Train the model on company documents
- ✅ Convert to GGUF format for inference
- ✅ Provide summary of created files

### 2. Manual Step-by-Step Process

#### Step 0: Download Base Model
```bash
python scripts\download_model.py
```
- Downloads model to `models/` directory for local usage.

#### Step 1: Prepare Training Data
```bash
python scripts\prepare_training_data_improved.py
```
- Scans `data/` folder for .md and .txt files
- Creates Q&A pairs from company documents
- Outputs `training_data.jsonl`

#### Step 2: Train Model
```bash
python scripts\train_model.py
```
- Trains Llama 3.2 1B Instruct on company data
- Saves model to `models/llama-3.2-1b-instruct-company-tuned/`

#### Step 3: Convert to GGUF
```bash
python scripts\convert_to_gguf_improved.py models\llama-3.2-1b-instruct-company-tuned
```
- Downloads llama.cpp automatically
- Converts to `models/llama-3.2-1b-instruct-company-tuned.gguf`

### 3. Test Your Trained Model

#### Start the API Server
```bash
python -m uvicorn app.main:app --reload
```

#### Run Automated Tests
```bash
python scripts\test_trained_model.py
```

#### Manual Testing via API
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company policy on leave?",
    "use_llm": true,
    "local_llm_model": "llama-3.2-1b-instruct-company-tuned"
  }'
```

## Sample Test Queries & Expected Responses

### 1. Leave Policy
**Query:** "What is the company policy on leave?"

**Expected Response Should Include:**
- Privilege Leave: 21 days annually
- Casual/Sick Leave: 12 days annually
- Wellness Days: 4 "Unplug" days per year
- Parental Leave: 26 weeks for primary caregivers

### 2. Working Hours
**Query:** "What are the core working hours?"

**Expected Response Should Include:**
- Core Hours: 11:00 AM to 3:00 PM local time
- Standard Week: 40 hours, Monday through Friday
- Flexible work schedule

### 3. Remote Work
**Query:** "Tell me about the remote work policy."

**Expected Response Should Include:**
- Remote-First for Saarthi Infotech and Vajra Solutions
- 3-days-in-office hybrid policy for Agni Pharma and Praxis
- Global presence across Mumbai, New York

### 4. Code of Conduct
**Query:** "What are the rules for moonlighting?"

**Expected Response Should Include:**
- Strictly prohibited
- Written consent required from HR and Legal
- No secondary employment without approval

## Troubleshooting

### Model Not Responding Correctly
1. **Check training data quality:**
   ```bash
   # Review the generated training data
   head -n 10 training_data.jsonl
   ```

2. **Increase training epochs:**
   - Edit `scripts/train_model.py`
   - Change `epochs=2` to `epochs=5`

3. **Add more training data:**
   - Add more .md/.txt files to `data/` folder
   - Re-run data preparation

### GGUF Conversion Fails
1. **Use HuggingFace model directly:**
   - The model works without GGUF conversion
   - GGUF is only for optimization

2. **Manual conversion:**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   python llama.cpp/convert_hf_to_gguf.py models/llama-3.2-1b-instruct-company-tuned --outfile models/llama-3.2-1b-instruct-company-tuned.gguf
   ```

### Server Connection Issues
1. **Check if server is running:**
   ```bash
   curl http://localhost:8000/api/models/list
   ```

2. **Check model availability:**
   - Ensure `models/llama-3.2-1b-instruct-company-tuned.gguf` exists
   - Or use HuggingFace format model

## File Structure After Training

```
lora_traning/
├── data/                          # Your company documents
├── models/
│   ├── llama-3.2-1b-instruct-company-tuned/  # HuggingFace format model
│   ├── llama-3.2-1b-instruct-company-tuned.gguf  # GGUF format (optimized)
│   └── llama-3.2-1b-instruct-company-tuned.json  # Training metadata
├── training_data.jsonl            # Prepared training data
└── scripts/
    ├── retrain_improved.bat       # Complete workflow
    ├── test_trained_model.py      # Automated testing
    └── ...
```

## Performance Expectations

### Good Performance Indicators:
- ✅ Model responds with company-specific information
- ✅ Mentions specific policies, numbers, and procedures
- ✅ Uses company terminology (Agni Holdings, Saarthi Infotech, etc.)
- ✅ Test script shows >60% theme matching

### Poor Performance Indicators:
- ❌ Generic responses not related to company
- ❌ Incorrect or made-up information
- ❌ Test script shows <30% theme matching
- ❌ Responses don't mention company policies

### Improvement Strategies:
1. **More Training Data:** Add more company documents
2. **Better Data Quality:** Ensure documents are well-structured
3. **Longer Training:** Increase epochs from 2 to 5-10
4. **Different Model:** Try a larger base model (requires more resources)

## Next Steps

1. **Production Deployment:**
   - Use the GGUF model for faster inference
   - Set up proper API authentication
   - Monitor model responses for accuracy

2. **Continuous Improvement:**
   - Regularly retrain with new company documents
   - Collect user feedback on responses
   - Fine-tune based on common queries

3. **Integration:**
   - Integrate with company chat systems
   - Create web interface for easy access
   - Set up automated retraining pipeline