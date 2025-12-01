# 🚀 Model Training Service Setup Guide

Complete guide to set up and run the model training service for fine-tuning models on company data.

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM (for training)
- Git (for GGUF conversion)
- Virtual environment activated

## 🔧 Installation & Setup

### 1. Install Dependencies

Create a requirements file for training dependencies:

```bash
# Create training requirements file
cat > requirements-training.txt << EOF
# Core training dependencies
transformers>=4.40.0
datasets>=2.14.0
torch>=2.0.0
accelerate>=0.24.0
peft>=0.6.0
bitsandbytes>=0.41.0

# Additional utilities
sentencepiece>=0.1.99
tokenizers>=0.22.0
safetensors>=0.4.3
huggingface-hub>=0.34.0
EOF

# Install training dependencies
pip install -r requirements-training.txt
```

### 2. Fix Transformers Version Issue

If you encounter `EncoderDecoderCache` import error:

```bash
# Upgrade transformers to latest version
pip install --upgrade transformers

# Verify the fix
python -c "from transformers import EncoderDecoderCache; print('✅ Import successful')"
```

### 3. Model Configuration

The training service uses **DistilGPT2** (open-source, no authentication required):

```python
# In model_training_service.py
self.model_name = "distilgpt2"  # Lightweight, open-source alternative
```

**Why DistilGPT2?**
- ✅ No authentication required (unlike Meta's Llama models)
- ✅ Lightweight and fast
- ✅ Well-supported by Hugging Face
- ✅ Production-ready

## 🏃‍♂️ Running Training

### Method 1: Using the Training Script

```bash
# Run the standalone training script
python scripts/train_model.py
```

### Method 2: Custom Training Parameters

```python
# In Python script or notebook
import asyncio
from app.services.model_training_service import train_company_model

async def custom_training():
    result = await train_company_model(
        output_name="my-custom-model",
        max_samples=500,  # Number of training samples
        epochs=2,         # Training epochs
        learning_rate=2e-5
    )
    print(f"Model saved to: {result['model_path']}")

# Run training
asyncio.run(custom_training())
```

## 📊 Training Process

The training follows these steps:

1. **Data Preparation** - Extracts documents from ChromaDB
2. **Data Filtering** - Excludes sensitive documents
3. **Format Conversion** - Creates instruction-tuning format
4. **Model Loading** - Downloads DistilGPT2 model
5. **Training** - Fine-tunes on company data
6. **Model Saving** - Saves in HuggingFace format
7. **GGUF Conversion** - Converts to GGUF for llama.cpp

## 🔄 GGUF Conversion

### Automatic Conversion (Recommended)

The training service attempts automatic GGUF conversion. If it fails, use manual conversion:

### Manual GGUF Conversion

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git temp_llama_cpp

# Convert to GGUF format
python temp_llama_cpp/convert_hf_to_gguf.py models/distilgpt2-company-tuned --outfile models/distilgpt2-company-tuned.gguf --outtype q8_0

# Clean up
rmdir /s /q temp_llama_cpp  # Windows
# rm -rf temp_llama_cpp     # Linux/Mac
```

### Available Quantization Types

- `f32` - Full precision (largest file)
- `f16` - Half precision
- `q8_0` - 8-bit quantization (recommended)
- `bf16` - Brain float 16
- `auto` - Automatic selection

## 📁 Output Structure

After training, you'll have:

```
models/
├── distilgpt2-company-tuned/          # HuggingFace format
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── distilgpt2-company-tuned.gguf      # GGUF format (89.5MB)
├── distilgpt2-company-tuned.json      # Training metadata
└── distilgpt2-company-tuned-training/ # Training checkpoints
```

## 🔍 Training Metadata

The `.json` file contains training information:

```json
{
  "model_name": "distilgpt2-company-tuned",
  "base_model": "distilgpt2",
  "training_samples": 53,
  "epochs": 2,
  "learning_rate": 2e-05,
  "trained_at": "2025-11-29T12:08:36.477920",
  "model_path": "models\\distilgpt2-company-tuned",
  "gguf_path": "models\\distilgpt2-company-tuned.gguf"
}
```

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 1. Import Error: `EncoderDecoderCache`
```bash
pip install --upgrade transformers>=4.40.0
```

#### 2. Gated Repository Error (Llama models)
- **Solution**: Use open-source models like `distilgpt2`
- **Alternative models**: `gpt2`, `microsoft/DialoGPT-small`

#### 3. Tokenization Error
- **Cause**: Incorrect data format
- **Solution**: Ensure proper tokenization settings (padding=False, return_tensors=None)

#### 4. GGUF Conversion Failed
```bash
# Manual conversion
git clone https://github.com/ggerganov/llama.cpp.git
python llama.cpp/convert_hf_to_gguf.py <model_path> --outfile <output.gguf> --outtype q8_0
```

#### 5. Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `fp16=True` for GPU training

## 📈 Training Tips

### Optimal Parameters

```python
# For small datasets (< 100 samples)
epochs=3
learning_rate=5e-5
max_samples=100

# For medium datasets (100-1000 samples)
epochs=2
learning_rate=2e-5
max_samples=500

# For large datasets (> 1000 samples)
epochs=1
learning_rate=1e-5
max_samples=1000
```

### Data Quality

- **Filter sensitive documents** - Exclude confidential data
- **Limit text length** - Use 512 tokens max
- **Balance departments** - Include diverse company content
- **Quality over quantity** - Better to have fewer high-quality samples

## 🚀 Using Trained Models

### In RAG System

The trained model can be used with your existing RAG system:

```python
# Use GGUF format with llama.cpp
model_path = "models/distilgpt2-company-tuned.gguf"

# Use HuggingFace format
model_path = "models/distilgpt2-company-tuned"
```

### With External Tools

- **Ollama**: Import GGUF file
- **LM Studio**: Load GGUF model
- **llama.cpp**: Direct GGUF usage
- **Text Generation WebUI**: HuggingFace format

## 📝 Quick Start Script

Create this script for easy retraining:

```bash
#!/bin/bash
# retrain.sh - Quick retraining script

echo "🚀 Starting model training..."

# Install/update dependencies
pip install --upgrade transformers>=4.40.0 datasets torch accelerate

# Run training
python scripts/train_model.py

# Convert to GGUF if needed
if [ ! -f "models/distilgpt2-company-tuned.gguf" ]; then
    echo "Converting to GGUF..."
    git clone https://github.com/ggerganov/llama.cpp.git temp_llama_cpp
    python temp_llama_cpp/convert_hf_to_gguf.py models/distilgpt2-company-tuned --outfile models/distilgpt2-company-tuned.gguf --outtype q8_0
    rm -rf temp_llama_cpp
fi

echo "✅ Training complete!"
echo "📁 HuggingFace model: models/distilgpt2-company-tuned/"
echo "📁 GGUF model: models/distilgpt2-company-tuned.gguf"
```

## 🔐 Security Notes

- **Data Privacy**: Training data comes from your ChromaDB
- **Model Security**: Trained models contain company information
- **Access Control**: Restrict access to trained models
- **Sensitive Data**: Automatically excludes highly confidential documents

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Model Training Best Practices](https://huggingface.co/docs/transformers/training)

---

**🎉 You're now ready to train custom models on your company data!**

For questions or issues, refer to the troubleshooting section or check the logs for detailed error messages.