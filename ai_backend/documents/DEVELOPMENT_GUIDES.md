# 🛠️ Development Guides & Technical Documentation

## 🔐 Authentication & Security Context

### JWT Authentication System

#### Token Creation
- **Function**: `create_access_token(user_data, session_id=None)`
- **Expiration**: Configurable via `JWT_EXPIRATION_DAYS`
- **Algorithm**: Configurable via `JWT_ALGORITHM`
- **Secret**: Secured via `JWT_SECRET_KEY`

#### Token Payload Structure
```json
{
  "user_id": "string",
  "username": "string", 
  "role": "string",
  "department": "string",
  "session_id": "string (optional)",
  "exp": "datetime",
  "iat": "datetime"
}
```

#### Authentication Endpoint
- **URL**: `POST /api/auth/token`
- **Purpose**: JWT login for role-based access

#### Security Features
- JWT token verification with expiration handling
- User action logging for token creation
- Security event logging for invalid/expired tokens
- Sensitive debug logging (remove in production)

#### Token Verification
- **Function**: `verify_token(token)`
- **Returns**: Decoded payload or None if invalid
- **Handles**: Expired tokens, invalid signatures, malformed tokens

## 🚀 Model Training Service Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for training)
- Git (for GGUF conversion)
- Virtual environment activated

### Installation & Setup

#### Install Training Dependencies
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

#### Fix Transformers Version Issue
```bash
# Upgrade transformers to latest version
pip install --upgrade transformers

# Verify the fix
python -c "from transformers import EncoderDecoderCache; print('✅ Import successful')"
```

### Model Configuration

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

### Running Training

#### Method 1: Using the Training Script
```bash
python scripts/train_model.py
```

#### Method 2: Custom Training Parameters
```python
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

### Training Process

1. **Data Preparation** - Extracts documents from ChromaDB
2. **Data Filtering** - Excludes sensitive documents
3. **Format Conversion** - Creates instruction-tuning format
4. **Model Loading** - Downloads DistilGPT2 model
5. **Training** - Fine-tunes on company data
6. **Model Saving** - Saves in HuggingFace format
7. **GGUF Conversion** - Converts to GGUF for llama.cpp

### GGUF Conversion

#### Automatic Conversion (Recommended)
The training service attempts automatic GGUF conversion.

#### Manual GGUF Conversion
```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git temp_llama_cpp

# Convert to GGUF format
python temp_llama_cpp/convert_hf_to_gguf.py models/distilgpt2-company-tuned --outfile models/distilgpt2-company-tuned.gguf --outtype q8_0

# Clean up
rmdir /s /q temp_llama_cpp  # Windows
```

#### Available Quantization Types
- `f32` - Full precision (largest file)
- `f16` - Half precision
- `q8_0` - 8-bit quantization (recommended)
- `bf16` - Brain float 16
- `auto` - Automatic selection

### Output Structure

After training:
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

### Training Metadata
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

## 🧬 LoRA Fine-Tuning Workflow

### What is LoRA
LoRA injects small, trainable rank-decomposition matrices into a frozen LLM so you can adapt behavior (policies, workflows, HR knowledge) with **far fewer parameters** and much less compute than full fine-tuning.

### Resource Requirements
- **Best**: GPU (fast, practical)
- **CPU-only**: Possible for small datasets but extremely slow
- **Workaround**: Fine-tune on Colab/GPU, use locally

### Proposed Approach

#### 1. Data Preparation
- Extract KBs into JSONL training pairs
- Format: `{"prompt": "<instruction>", "response": "<desired reply>"}`
- Place in `/data/lora_training.jsonl`

#### 2. Choose Fine-tune Target
- Preferred: Smaller local model for CPU inference
- Alternative: Mistral7B (requires GPU or very slow CPU)

#### 3. Tooling
- Use `transformers` + `peft` (LoRA) + `accelerate`
- CPU training: small batch sizes, gradient accumulation

#### 4. Training Script Concept
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

MODEL = "mistral/your-hf-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)

lora_config = LoraConfig(r=8, alpha=32, target_modules=["q_proj","v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)

# Training hyperparams for small dataset:
# epochs: 3–5, batch_size: 1–4 (CPU), learning_rate: 1e-4 ~ 3e-4
# lora_r: 8, lora_alpha: 32

# Save adapter (peft format)
adapter_dir = "adapter-save"
model.save_pretrained(adapter_dir)
```

#### 5. Integration Options

**Option A: Local inference via PyTorch + Transformers + PEFT**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "mistral/your-hf-base"
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
adapter_path = "app/models/lora/<adapter_name>"
model = PeftModel.from_pretrained(base_model, adapter_path)
```

**Option B: Merge adapter → GGUF for llama.cpp**
```python
# Merge adapter into base weights
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
peft = PeftModel.from_pretrained(base, "adapter-save")
peft.merge_and_unload()
base.save_pretrained("merged-model")
```

### Colab Training + Local Inference

#### Train on Colab (GPU)
```bash
# Install on Colab
pip install transformers accelerate peft datasets safetensors

# Train and save adapter
# Download adapter folder to local project
```

#### Use Locally
```python
# Load base + adapter locally
model = PeftModel.from_pretrained(base_model, "app/models/lora/<adapter_name>")
```

## 🔧 Troubleshooting Common Issues

### Training Service Issues

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

### Training Tips

#### Optimal Parameters
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

#### Data Quality Guidelines
- **Filter sensitive documents** - Exclude confidential data
- **Limit text length** - Use 512 tokens max
- **Balance departments** - Include diverse company content
- **Quality over quantity** - Better to have fewer high-quality samples

### Quick Retraining Script
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
```

## 🔒 Security Notes

### Data Privacy
- Training data comes from your ChromaDB
- Trained models contain company information
- Restrict access to trained models
- Automatically excludes highly confidential documents

### Access Control
- Use RBAC for training permissions
- Log all training activities
- Validate data sources before training
- Implement approval workflows for model deployment

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Model Training Best Practices](https://huggingface.co/docs/transformers/training)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

This guide provides comprehensive coverage of authentication, model training, LoRA fine-tuning, and troubleshooting for the Multi-Provider Enterprise RAG System development.