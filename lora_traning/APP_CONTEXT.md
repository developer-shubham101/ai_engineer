# 🚀 LoRA Training and Testing System - Technical Context

**Single Source of Truth for AI Assistant Understanding**

> **For AI Assistants**: This file contains complete system architecture, API specifications, and implementation details. Use this as primary context for code generation, debugging, and system understanding.

---

## 1. System Overview

### Purpose
A **minimal LoRA fine-tuning and testing system** for training custom models on company data and testing their performance. Focused on model training workflows without RAG complexity.

### Supported Models
- **Local Models**: GGUF format models loaded via llama-cpp-python
- **Training**: LoRA fine-tuning on Llama 3.2 1B base model
- **Output**: Trained models saved in HuggingFace format + GGUF conversion

### Key Features
- ✅ **LoRA fine-tuning** with company data
- ✅ **Model testing** via simple query endpoint
- ✅ **Model management** and listing
- ✅ **Background training jobs** with progress tracking
- ✅ **GGUF model loading** for inference testing

---

## 2. Architecture Overview

```
User Request → FastAPI Router → Service Layer → Model/Training Response
                                      ↓
                              Core Components:
                              • Model Manager (GGUF Loading)
                              • Training Service (LoRA Fine-tuning)
                              • Local Model Manager (Model Discovery)
```

### Core Components

**⚙️ Services**
- `model_manager.py` - GGUF model loading & caching
- `local_model_manager.py` - Model detection and listing
- `model_training_service.py` - LoRA fine-tuning service
- `logging_config.py` - Basic logging setup

**🌐 API Layer**
- `main.py` - FastAPI application with lifecycle management
- `api_routes_rag.py` - Simple query endpoint for model testing
- `api_routes_models.py` - Model management endpoints
- `api_routes_training.py` - Model training endpoints

---

## 3. Directory Structure

```
lora_traning/
├── app/
│   ├── services/              # Core business logic
│   │   ├── __init__.py
│   │   ├── model_training_service.py # LoRA fine-tuning
│   │   ├── model_manager.py         # GGUF model loading
│   │   └── local_model_manager.py   # Model discovery
│   ├── config/               # Configuration files
│   │   └── local_models.json       # Model configurations
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   ├── api_routes_rag.py     # Simple query endpoint
│   ├── api_routes_models.py  # Model management
│   ├── api_routes_training.py # Training endpoints
│   └── logging_config.py    # Logging setup
├── models/                  # Local LLM files (GGUF + trained models)
│   ├── distilgpt2-company-tuned/     # HuggingFace format
│   ├── distilgpt2-company-tuned.gguf # GGUF format
│   └── distilgpt2-company-tuned.json # Training metadata
├── data/                   # Training data (text files)
│   └── company/            # Company documents
├── raw_data/               # Original markdown files
│   └── company/            # Source documents (.md format)
├── scripts/                # Training and conversion scripts
│   ├── doc_parser.py
│   ├── train_model.py
│   ├── convert_to_gguf_improved.py
│   ├── test_trained_model.py
│   └── retrain_improved.bat
├── logs/                   # Application logs
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── APP_CONTEXT.md         # This file
├── DOCUMENT_PARSING.md    # Document parsing guide
├── FIXES_AND_IMPROVEMENTS.md # Known issues and fixes
└── USAGE_GUIDE.md         # Usage instructions
```

---

## 4. API Endpoints

### Simple Query (`/api/`)

**POST /api/query** - Test trained models
```json
Request: {
  "question": "string",
  "use_llm": false,
  "max_tokens": 256,
  "debug": false,
  "local_llm_model": "string"  // Optional model name
}

Response: {
  "answer": "string",
  "final_prompt": "string"  // Debug: actual prompt sent to LLM
}
```

### Model Management (`/api/models/`)

**GET /api/models/list** - List available models
**POST /api/models/refresh** - Refresh model cache

### Model Training (`/api/training/`)

**GET /api/training/status** - Check training availability
**POST /api/training/start** - Start LoRA training job
**GET /api/training/jobs/{id}** - Monitor training progress
**GET /api/training/models** - List trained models
**DELETE /api/training/models/{name}** - Delete trained model

---

## 5. Training System

### Model Configuration
- **Base Model**: Configurable via `app/config/model_config.py` (default: DistilGPT2)
- **Training Format**: "Question: X\nAnswer: Y"
- **Context Length**: 512 tokens
- **Training Epochs**: 2-3 (adjustable)

### Training Process
1. **Document Conversion**: Convert .md/.html/.txt to plain text
2. **Data Preparation**: Create Q&A pairs from company documents
3. **Format Conversion**: Create instruction-tuning format
4. **Model Training**: Fine-tune DistilGPT2 on company data
5. **Export**: Save in HuggingFace format
6. **GGUF Conversion**: Convert for llama.cpp inference

### Supported Document Formats
- ✅ **Markdown (.md)** - Company policies, documentation
- ✅ **HTML (.html, .htm)** - Web pages, exported documents
- ✅ **Plain Text (.txt)** - Simple text files
- 🔄 **PDF (.pdf)** - Coming soon
- 🔄 **Word (.docx)** - Coming soon

### Training Request
```json
{
  "output_name": "auto-generated-from-model-name",
  "base_model": "configurable-via-model_config.py",
  "max_samples": 500,
  "epochs": 2,
  "learning_rate": 2e-5
}
```

---

## 6. Model Testing

### Query Format
- **Training format**: `Question: {question}\nAnswer:`
- **Stop tokens**: `["Question:", "\n\n"]`
- **Temperature**: 0.7 for balanced creativity
- **Context length**: 512 tokens

### Model Loading
- **GGUF format**: Optimized for CPU inference
- **Caching**: Models cached in memory for repeated use
- **Auto-discovery**: Scans `models/` directory for available models
- **Fallback**: Uses first available model if none specified

### Testing Workflow
1. **Convert documents** using `doc_parser.py`
2. **Prepare training data** using `prepare_training_data_improved.py`
3. **Train model** using `train_model.py`
4. **Convert to GGUF** using `convert_to_gguf_improved.py`
5. **Test model** using automated test script

### Sample Test Queries & Expected Responses

#### Leave Policy
**Query**: "What is the company policy on leave?"
**Expected Response**: Should include:
- Privilege Leave: 21 days annually
- Casual/Sick Leave: 12 days annually
- Wellness Days: 4 "Unplug" days per year
- Parental Leave: 26 weeks for primary caregivers

#### Working Hours
**Query**: "What are the core working hours?"
**Expected Response**: Should include:
- Core Hours: 11:00 AM to 3:00 PM local time
- Standard Week: 40 hours, Monday through Friday
- Flexible work schedule

#### Remote Work
**Query**: "Tell me about the remote work policy."
**Expected Response**: Should include:
- Remote-First for Saarthi Infotech and Vajra Solutions
- 3-days-in-office hybrid policy for Agni Pharma and Praxis
- Global presence across Mumbai, New York

#### Code of Conduct
**Query**: "What are the rules for moonlighting?"
**Expected Response**: Should include:
- Strictly prohibited
- Written consent required from HR and Legal
- No secondary employment without approval

### Document Processing System
- **Multi-Format Support**: .md, .html, .txt files with nested directories
- **Automatic Conversion**: Clean text extraction from various formats
- **Quality Control**: Content length limits and formatting cleanup
- **Structure Preservation**: Maintains directory hierarchy in converted files

### Training Data Generation
- **Smart Q&A Creation**: Multiple question variations per document section
- **Company-Specific Templates**: Policy-focused question generation
- **Content Optimization**: Length limits and quality filtering
- **Comprehensive Coverage**: Document-level and section-level training pairs

### Model Training System
- **DistilGPT2 Fine-tuning**: Lightweight model for fast training
- **Automated Pipeline**: Complete workflow from documents to GGUF
- **Quality Assurance**: Automated testing with performance scoring
- **Format Conversion**: HuggingFace to GGUF conversion for efficient inference
- **Progress Tracking**: Detailed logging and error handling

---

## 7. Quick Start Guide

### Complete Workflow (Recommended)
```bash
# Run everything in one command
scripts\retrain_improved.bat
```

### Test Your Model
```bash
# Start server
python -m uvicorn app.main:app --reload

# Run tests
python scripts\test_trained_model.py
```

### Manual Query
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company policy on leave?",
    "use_llm": true,
    "local_llm_model": "distilgpt2-company-tuned"
  }'
```

---

## 8. Dependencies

### Core Requirements
```python
# Core FastAPI and server
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Machine Learning and Training
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0

# Model Inference
llama-cpp-python>=0.2.0

# Document Parsing
markdown>=3.4.0
beautifulsoup4>=4.12.0

# Utilities
python-multipart>=0.0.6
pytest
```

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB+ (16GB recommended for training)
- **Storage**: 10GB+ for models and data
- **GPU**: Optional (CUDA-compatible for faster training)

### Model Storage
- **Base models**: Downloaded to `models/` directory
- **Trained models**: Saved as subdirectories in `models/`
- **GGUF files**: Quantized models for inference
- **Training data**: Text files in `data/` directory

---

## 8. Core Service Functions

### Document Parser
```python
def parse_document(file_path: Path) -> str:
    # Convert .md/.html/.txt to plain text
    # Remove formatting and tags
    # Return clean text content

def scan_and_convert(data_dir: Path) -> List[Dict]:
    # Recursively scan for supported files
    # Convert all to text format
    # Maintain directory structure
```

### Training Data Preparation
```python
def prepare_training_data(max_samples: int) -> List[Dict]:
    # Read from data/ or data/converted/
    # Create Q&A pairs with multiple question variations
    # Format for instruction tuning

def generate_questions_for_topic(topic: str) -> List[str]:
    # Generate company-specific questions
    # Multiple variations per topic
    # Return formatted questions
```

### ModelTrainingService
```python
def train_model(output_name: str, base_model: str, max_samples: int, epochs: int) -> Dict:
    # 1. Prepare training data from documents
    # 2. Load configurable base model and tokenizer
    # 3. Train model with company data
    # 4. Save and convert to GGUF
    # 5. Return training metadata
```

### ModelManager
```python
def load_llama_model(model_name: str) -> Optional[Llama]:
    # Load GGUF model with llama-cpp-python
    # Cache loaded models
    # Return Llama instance for inference

def get_model(model_name: str) -> Optional[ModelInfo]:
    # Get model information by name
    # Scan models directory
    # Return model metadata
```

### LocalModelManager
```python
def get_available_models() -> List[Dict]:
    # Scan models directory
    # Detect GGUF files and trained models
    # Return list of available models

def refresh_models() -> List[ModelInfo]:
    # Refresh model cache
    # Detect new models
    # Update model registry
```

---

## 9. Data Models

### QueryRequest
```python
{
  "question": str,
  "use_llm": bool,
  "max_tokens": int,
  "debug": bool,
  "local_llm_model": Optional[str]
}
```

### TrainingRequest
```python
{
  "output_name": str,
  "base_model": str,
  "max_samples": int,
  "epochs": int,
  "learning_rate": float
}
```

### TrainingJob
```python
{
  "job_id": str,
  "status": str,  # starting, training, completed, failed
  "progress": float,
  "created_at": str,
  "model_name": Optional[str],
  "error": Optional[str]
}
```

### ModelInfo
```python
{
  "name": str,
  "path": str,
  "type": str,  # "base" or "lora"
  "size": Optional[int]
}
```

---

## 10. Automated Workflow

### Complete Training Pipeline
```bash
# One command for complete workflow
scripts\retrain_improved.bat
```

**Workflow Steps:**
1. **Document Conversion** - Convert .md/.html to text
2. **Data Preparation** - Create Q&A pairs
3. **Dependency Installation** - Install required packages
4. **Model Training** - Train on company data
5. **GGUF Conversion** - Convert for inference
6. **Summary Report** - Show available files

### Manual Steps
```bash
# Step-by-step execution
python scripts\doc_parser.py
python scripts\prepare_training_data_improved.py
python scripts\train_model.py
python scripts\convert_to_gguf_improved.py models\distilgpt2-company-tuned
```

### Testing
```bash
# Start API server
python -m uvicorn app.main:app --reload

# Run automated tests
python scripts\test_trained_model.py
```

## 11. Current Project Structure

```
lora_traning/
├── app/                     # FastAPI application
│   ├── services/           # Core business logic
│   ├── config/            # Configuration files
│   ├── main.py            # FastAPI app entry point
│   └── api_routes_*.py    # API endpoints
├── data/                   # Processed training data (text files)
│   └── company/           # Company documents (.txt format)
├── raw_data/              # Original source documents
│   └── company/           # Source documents (.md format)
├── models/                # Trained models and GGUF files
│   ├── distilgpt2-company-tuned/     # HuggingFace format
│   ├── distilgpt2-company-tuned.gguf # GGUF format
│   └── distilgpt2-company-tuned.json # Training metadata
├── scripts/               # Training and utility scripts
│   ├── doc_parser.py              # Document conversion
│   ├── train_model.py             # Model training
│   ├── convert_to_gguf_improved.py # GGUF conversion
│   ├── test_trained_model.py      # Automated testing
│   └── retrain_improved.bat       # Complete workflow
├── logs/                  # Application logs
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── APP_CONTEXT.md        # This technical context file
├── DOCUMENT_PARSING.md   # Document parsing documentation
├── FIXES_AND_IMPROVEMENTS.md # Known issues and solutions
└── USAGE_GUIDE.md        # User guide and instructions
```

## 12. Model Configuration Management

### Centralized Configuration
- **File**: `app/config/model_config.py`
- **Purpose**: Single source of truth for model settings
- **Usage**: Import `ModelConfig` class to get default settings

### Adding New Models
1. Update `DEFAULT_BASE_MODEL` in `model_config.py`
2. Update `DEFAULT_OUTPUT_NAME` if needed
3. All scripts automatically use new configuration

### Performance Expectations

### Good Performance (>60% test score):
- ✅ Company-specific responses
- ✅ Accurate policy information
- ✅ Proper terminology (Agni Holdings, etc.)
- ✅ Specific numbers and procedures

### Poor Performance (<30% test score):
- ❌ Generic responses
- ❌ Incorrect information
- ❌ No company context

### Improvement Strategies:
- Add more company documents
- Increase training epochs (2→5)
- Improve data quality
- Use larger base model

## Directory Paths
```python
MODELS_DIR = Path("models")      # GGUF models and trained models
DATA_DIR = Path("data")          # Training data (text files)
```

### Training Settings
```python
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_BATCH_SIZE = 1
MAX_TRAINING_SAMPLES = 1000
```

### Alternative Models
```python
# Open-source alternatives (no authentication required)
"distilgpt2"              # Lightweight, fast
"gpt2"                    # Standard GPT-2
"microsoft/DialoGPT-small" # Dialog-focused
```

### GGUF Quantization Types
```python
"f32"   # Full precision (largest file)
"f16"   # Half precision
"q8_0"  # 8-bit quantization (recommended)
"bf16"  # Brain float 16
"auto"  # Automatic selection
```

### Model Loading
```python
DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_THREADS = 4
DEFAULT_TEMPERATURE = 0.7
STOP_TOKENS = ["<|end|>", "<|user|>"]
```

---

## 11. Usage Examples

### Start Training
```bash
curl -X POST "/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "output_name": "my-custom-model",
    "base_model": "meta-llama/Llama-3.2-1B",
    "max_samples": 500,
    "epochs": 3
  }'
```

### Monitor Training
```bash
curl "/api/training/jobs/{job_id}"
```

### Test Model
```bash
curl -X POST "/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is our company policy?",
    "use_llm": true,
    "local_llm_model": "my-custom-model",
    "debug": true
  }'
```

### List Models
```bash
curl "/api/models/list"
```

---

## 12. Deployment

### Development
```bash
uvicorn app.main:app --reload --port 5444
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5444
```

### Requirements Installation
```bash
pip install -r requirements.txt
```

### Directory Setup
```bash
mkdir models data
# Place training data in data/ directory
# Models will be saved to models/ directory
```

---

## 13. Troubleshooting

### Common Issues & Solutions

#### 1. Import Error: `EncoderDecoderCache`
```bash
pip install --upgrade transformers>=4.40.0
```

#### 2. Out of Memory During Training
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `fp16=True` for GPU training
- Reduce `max_samples` parameter

#### 3. GGUF Conversion Failed
```bash
# Manual conversion
git clone https://github.com/ggerganov/llama.cpp.git temp_llama_cpp
python temp_llama_cpp/convert_hf_to_gguf.py models/my-model --outfile models/my-model.gguf --outtype q8_0
rmdir /s /q temp_llama_cpp  # Windows
```

#### 4. Model Loading Error
- Verify GGUF file exists in `models/` directory
- Check file permissions and disk space
- Ensure model name matches file/directory name

#### 5. Training Data Not Found
- Place `.txt` files in `data/` directory
- Verify file encoding is UTF-8
- Check file permissions

### Training Tips

#### Optimal Parameters
```python
# Small datasets (< 100 samples)
epochs=3, learning_rate=5e-5, max_samples=100

# Medium datasets (100-1000 samples)  
epochs=2, learning_rate=2e-5, max_samples=500

# Large datasets (> 1000 samples)
epochs=1, learning_rate=1e-5, max_samples=1000
```

#### Data Quality Guidelines
- **Filter sensitive content** - Exclude confidential data
- **Limit text length** - Use 512 tokens max per sample
- **Balance content** - Include diverse company information
- **Quality over quantity** - Fewer high-quality samples work better

---

## 14. AI Assistant Instructions

**When working with this system:**

1. **Focus on training workflow** - This is a LoRA fine-tuning system, not RAG
2. **Use minimal dependencies** - Only include what's needed for training and testing
3. **Handle training data** - Read from `data/` directory, format for instruction tuning
4. **Manage models** - GGUF loading, caching, and model discovery
5. **Background jobs** - Training runs asynchronously with progress tracking
6. **Simple testing** - Direct model inference without retrieval
7. **Error handling** - Use FastAPI `HTTPException` with proper status codes
8. **Type hints** - Include annotations for all parameters and returns
9. **Logging** - Basic logging for training progress and errors
10. **Model formats** - Support both HuggingFace and GGUF formats

**When debugging:**
- Check if training data exists in `data/` directory
- Verify model files are in correct format (GGUF for inference)
- Ensure sufficient disk space for model training and storage
- Check CUDA availability for GPU training
- Validate model loading and caching mechanisms
- Review troubleshooting section for common issues

**When adding features:**
- Keep the system minimal and focused on training/testing
- Maintain compatibility with existing model formats
- Add appropriate error handling and logging
- Follow the established service layer pattern
- Include comprehensive type hints and documentation

---



---



---

---

## Quick Model Change Guide

### To Add/Change Models (One Place Only):

**File**: `app/config/model_config.py`

```python
# Change these two lines only:
DEFAULT_BASE_MODEL = "your-new-model"  # e.g., "gpt2", "microsoft/DialoGPT-small"
DEFAULT_OUTPUT_NAME = "your-new-model-company-tuned"  # Auto-generated if not specified
```

**Available Models**:
- `distilgpt2` - Fast, lightweight (default)
- `gpt2` - Standard GPT-2
- `microsoft/DialoGPT-small` - Dialog-focused
- `microsoft/DialoGPT-medium` - Larger dialog model

**Test Configuration**: Run `python scripts/change_model_example.py`

---

**Last Updated**: 2025-01-10
**Project Structure Synced**: 2025-01-10

This context file provides complete system understanding for the LoRA training and testing system. The system is designed to be minimal, focused, and efficient for model fine-tuning workflows.