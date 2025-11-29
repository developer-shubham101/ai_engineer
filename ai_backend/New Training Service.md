# 🎯 Model Training Service

> **Fine-tune local models on your company documents with enterprise security**

Extend your RAG system with custom-trained models that understand your specific domain and terminology. Train Llama 3.2 1B on filtered company documents with automatic format conversion and progress tracking.

## ✨ Key Features

- 🤖 **Local Model Fine-tuning** - Train Llama 3.2 1B on company documents
- 🔒 **Data Security** - Automatic filtering of sensitive documents
- 📦 **Dual Format Export** - HuggingFace and GGUF formats
- 🚀 **Background Processing** - Non-blocking API with job tracking
- 🛡️ **RBAC Integration** - SuperAdmin-only training access
- 📊 **Progress Monitoring** - Real-time training status updates

## 🏗️ Architecture

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Training Service** | Fine-tuning logic | `app/services/model_training_service.py` |
| **Training API** | REST endpoints | `app/api_routes_training.py` |
| **Background Jobs** | Async training | Built-in job tracking |
| **Format Converter** | GGUF export | Automatic conversion |

### Training Pipeline

```mermaid
graph LR
    A[ChromaDB Documents] --> B[Data Filtering]
    B --> C[Instruction Format]
    C --> D[Fine-tuning]
    D --> E[HuggingFace Export]
    E --> F[GGUF Conversion]
    F --> G[Model Ready]
```

## 🚀 Quick Start

### Installation

```bash
# Install training dependencies
pip install -r requirements-training.txt

# Or install manually
pip install transformers datasets torch accelerate peft bitsandbytes
```

### Start Training

```bash
# Check training availability
curl -X GET "http://localhost:8000/api/training/status"

# Start training (SuperAdmin token required)
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "output_name": "llama-3.2-1b-company-tuned",
    "max_samples": 1000,
    "epochs": 3,
    "learning_rate": 2e-5
  }'
```

### Standalone Training

```bash
# Run training script directly
python scripts/train_model.py
```

## 🔌 API Reference

### Training Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/training/status` | GET | Check training availability | No |
| `/api/training/start` | POST | Start training job | SuperAdmin |
| `/api/training/jobs/{id}` | GET | Check training progress | Yes |
| `/api/training/models` | GET | List trained models | Yes |
| `/api/training/models/{name}` | DELETE | Delete trained model | SuperAdmin |

### Start Training Request

```json
{
  "output_name": "llama-3.2-1b-company-tuned",
  "max_samples": 1000,
  "epochs": 3,
  "learning_rate": 2e-5,
  "batch_size": 4,
  "max_length": 512
}
```

### Training Status Response

```json
{
  "job_id": "train_123",
  "status": "running",
  "progress": 0.45,
  "current_epoch": 2,
  "total_epochs": 3,
  "estimated_time_remaining": "15 minutes"
}
```

## 📁 Output Structure

After successful training:

```
models/
├── llama-3.2-1b-company-tuned/          # HuggingFace format
│   ├── config.json                       # Model configuration
│   ├── pytorch_model.bin                 # Trained weights
│   ├── tokenizer.json                    # Tokenizer config
│   └── training_args.json                # Training parameters
├── llama-3.2-1b-company-tuned.gguf      # GGUF for llama.cpp
└── llama-3.2-1b-company-tuned.json      # Training metadata
```

### Model Metadata

```json
{
  "model_name": "llama-3.2-1b-company-tuned",
  "base_model": "meta-llama/Llama-3.2-1B-Instruct",
  "training_date": "2024-01-15T10:30:00Z",
  "documents_used": 1000,
  "training_parameters": {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 4
  },
  "performance_metrics": {
    "final_loss": 0.85,
    "training_time": "45 minutes"
  }
}
```

## 🔐 Security & Data Filtering

### Document Filtering Rules

- ❌ **Excluded**: `super_confidential`, `highly_confidential` documents
- ✅ **Included**: `public`, `internal`, `department_confidential` documents
- 🔍 **Automatic**: Metadata-based filtering during data preparation

### Access Control

- **Training Operations**: SuperAdmin role required
- **Model Listing**: Any authenticated user
- **Model Deletion**: SuperAdmin role required
- **Progress Monitoring**: User who started the job + SuperAdmin

## ⚙️ Configuration

### Training Parameters

```python
# Default training configuration
TRAINING_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "max_samples": 1000,
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "max_length": 512,
    "warmup_steps": 100,
    "save_steps": 500
}
```

### Hardware Requirements

- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB VRAM
- **Training Time**: ~30-60 minutes (1000 samples, 3 epochs)

## 📊 Monitoring & Optimization

### Training Metrics

- **Loss Tracking**: Monitor training and validation loss
- **Progress Updates**: Real-time epoch and step progress
- **Resource Usage**: Memory and GPU utilization
- **Time Estimates**: Remaining training time

### Performance Tips

- **Batch Size**: Adjust based on available VRAM
- **Learning Rate**: Start with 2e-5, adjust if needed
- **Sample Size**: More samples = better performance, longer training
- **Epochs**: 3-5 epochs typically sufficient

## 🛠️ Advanced Usage

### Custom Training Scripts

```python
from app.services.model_training_service import ModelTrainingService

# Initialize training service
trainer = ModelTrainingService()

# Start custom training
job_id = await trainer.start_training(
    output_name="custom-model",
    max_samples=2000,
    epochs=5,
    learning_rate=1e-5
)

# Monitor progress
status = await trainer.get_job_status(job_id)
print(f"Training progress: {status['progress']:.1%}")
```

### Integration with RAG System

```bash
# Use trained model in RAG queries
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are our company policies?",
    "model_name": "llama-3.2-1b-company-tuned",
    "use_llm": true
  }'
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Out of Memory** | Reduce batch_size or max_length |
| **Slow Training** | Increase batch_size if memory allows |
| **Poor Performance** | Increase epochs or sample size |
| **GGUF Conversion Fails** | Check llama-cpp-python installation |

### Debug Mode

```bash
# Enable detailed logging
export TRAINING_DEBUG=true
python scripts/train_model.py
```

---

**Ready to train your custom models? Start with the API endpoints above or run the standalone script for quick testing.**