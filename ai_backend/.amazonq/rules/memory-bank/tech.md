# Technology Stack & Dependencies

## Programming Languages
- **Python 3.8+** - Primary language for all components
- **JavaScript/JSON** - Configuration files and API schemas
- **Markdown** - Documentation and sample data
- **SQL** - Database queries and schema definitions

## Core Framework
- **FastAPI 0.104+** - Modern web framework with automatic OpenAPI documentation
- **Uvicorn** - ASGI server for FastAPI applications
- **Pydantic** - Data validation and settings management
- **SQLAlchemy** - Database ORM and connection management

## LLM & AI Libraries

### Local Model Support
- **llama-cpp-python** - Local LLM inference with GGUF format support
- **sentence-transformers** - Embedding model management
- **transformers** - Hugging Face model integration
- **torch** - PyTorch for model operations

### Cloud Provider SDKs
- **openai** - OpenAI GPT API integration
- **google-generativeai** - Google Gemini API integration
- **requests** - HTTP client for API calls

## Vector Databases
- **chromadb** - Primary vector database for embeddings
- **faiss-cpu** - Alternative vector similarity search
- **numpy** - Numerical operations for embeddings

## Authentication & Security
- **python-jose[cryptography]** - JWT token handling
- **passlib[bcrypt]** - Password hashing and verification
- **python-multipart** - Form data handling
- **cryptography** - Cryptographic operations

## Multimodal Processing
- **Pillow (PIL)** - Image processing and manipulation
- **opencv-python** - Computer vision operations
- **vosk** - Speech-to-text processing
- **pydub** - Audio file manipulation
- **pytesseract** - OCR text extraction

## Data Processing
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning utilities
- **joblib** - Model serialization and parallel processing
- **python-docx** - Word document processing
- **PyPDF2** - PDF document parsing

## Development & Testing
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support
- **httpx** - Async HTTP client for testing
- **black** - Code formatting
- **flake8** - Code linting

## Configuration & Environment
- **python-dotenv** - Environment variable management
- **pyyaml** - YAML configuration parsing
- **toml** - TOML configuration support

## Build & Deployment

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Install multimodal features
pip install -r requirements_multimodal.txt

# Install agent capabilities
pip install -r requirements_agents.txt

# Start development server
python -m app.main

# Run tests
python -m pytest tests/

# Run specific test module
python -m pytest test_module/
```

### Docker Support
```bash
# Build container
docker build -t ai-backend .

# Run with docker-compose
docker-compose up -d
```

### Model Management
```bash
# Download embedding models
python scripts/download_embeddings_models.py

# Download HuggingFace models
python scripts/download_hf_model.py

# Convert models to GGUF
python scripts/convert_to_gguf.py
```

## Environment Configuration

### Required Environment Variables
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model Settings
DEFAULT_MODEL_NAME=mistral-7b-instruct-v0.2
EMBEDDING_MODEL_NAME=bge-small-en-v1.5

# Optional: Cloud Provider API Keys
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_TOKEN=your_hf_token
```

### Model File Structure
```
models/
├── mistral-7b-instruct-v0.2.Q3_K_M.gguf
├── phi-2-q4_k_m.gguf
├── llama-3.2-1b-instruct-q4_k_m.gguf
├── gemma-2b-it-q4_k_m.gguf
└── qwen2-1.5b-instruct-q4_k_m.gguf

embeddings_models/
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

## Database Systems
- **SQLite** - Local database for users, sessions, conversations
- **ChromaDB** - Vector database with SQLite backend
- **FAISS** - In-memory vector index with pickle serialization

## API & Documentation
- **OpenAPI 3.0** - Automatic API documentation generation
- **Swagger UI** - Interactive API documentation at `/docs`
- **ReDoc** - Alternative API documentation at `/redoc`

## Performance & Monitoring
- **Python Logging** - Structured logging with multiple levels
- **File-based Logging** - Separate logs for debug, application, security
- **Token Usage Tracking** - Monitor LLM API costs and efficiency
- **Response Time Metrics** - Performance monitoring across providers

## Platform Support
- **Windows** - Primary development platform with batch scripts
- **Linux/macOS** - Cross-platform compatibility
- **Docker** - Containerized deployment support

## Version Requirements
```
Python >= 3.8
FastAPI >= 0.104
ChromaDB >= 0.4.0
llama-cpp-python >= 0.2.0
sentence-transformers >= 2.2.0
openai >= 1.0.0
```

## Optional Dependencies
- **PDF Processing**: PyPDF2, pdfplumber
- **Audio Processing**: vosk, pydub, soundfile
- **Vision Processing**: opencv-python, pytesseract
- **Agent Framework**: crewai, langchain
- **Advanced ML**: scikit-learn, joblib