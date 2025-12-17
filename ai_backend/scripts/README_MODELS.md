# 🤖 Model Download & Installation Guide

This directory contains scripts for downloading and managing AI models for the multimodal RAG system.

## 📁 Files

- `download_hf_model.py` - Enhanced model downloader (LLM + Multimodal)
- `install_multimodal.py` - Automated multimodal dependencies installer
- `install_multimodal.bat` - Windows batch installer
- `README_MODELS.md` - This guide

## 🚀 Quick Start

### 1. List Available Models
```bash
# List all models (LLM + Multimodal)
python download_hf_model.py --list

# List only LLM models
python download_hf_model.py --list --type llm

# List only multimodal models
python download_hf_model.py --list --type multimodal
```

### 2. Install Multimodal Dependencies
```bash
# Automated installation (recommended)
python install_multimodal.py

# Windows batch installer
install_multimodal.bat

# Manual installation commands
python download_hf_model.py --install-deps
```

### 3. Download Models

#### LLM Models (GGUF)
```bash
# Download recommended model
python download_hf_model.py --download phi2

# Download specific model
python download_hf_model.py --download llama32-1b

# Download all LLM models
python download_hf_model.py --all --type llm
```

#### Multimodal Models
```bash
# Download Vosk STT model
python download_hf_model.py --download vosk-small-en --type multimodal

# Download all downloadable multimodal models
python download_hf_model.py --all --type multimodal
```

## 📊 Model Categories

### 🤖 LLM Models (GGUF Format)
| Model | Size | Status | Description |
|-------|------|--------|-------------|
| `phi2` | 2.7B | ⭐ Recommended | High-quality reasoning model |
| `llama32-1b` | 1B | Available | Efficient edge model |
| `llama32-3b` | 3B | Available | Balanced performance |
| `gemma-2b` | 2B | Available | Safety-aligned model |
| `qwen2-1.5b` | 1.5B | Available | Multilingual model |
| `mistral-7b` | 7B | Available | Production default |

### 🎙️ Speech-to-Text Models
| Model | Size | Type | Description |
|-------|------|------|-------------|
| `vosk-small-en` | 40MB | ⭐ Download | Lightweight English STT |
| `vosk-en` | 1.8GB | Download | Full accuracy English STT |
| `whisper-tiny` | 39MB | Auto-download | Fastest Whisper |
| `whisper-base` | 74MB | ⭐ Auto-download | Balanced Whisper |
| `whisper-small` | 244MB | Auto-download | Accurate Whisper |

### 🔊 Text-to-Speech Models
| Model | Type | Installation |
|-------|------|-------------|
| `pyttsx3` | ⭐ System | `pip install pyttsx3` |
| `espeak` | System | System package required |

### 👁️ Vision/OCR Models
| Model | Type | Installation |
|-------|------|-------------|
| `tesseract` | ⭐ System | System package + `pip install pytesseract` |
| `paddleocr` | Package | `pip install paddlepaddle paddleocr` |

## 🛠️ Advanced Usage

### Force Re-download
```bash
python download_hf_model.py --download phi2 --force
```

### Scan Existing Models
```bash
python download_hf_model.py --scan
```

### Download All Models
```bash
# Download everything
python download_hf_model.py --all

# Download only LLM models
python download_hf_model.py --all --type llm

# Download only multimodal models
python download_hf_model.py --all --type multimodal
```

## 📦 Installation Requirements

### Python Packages
```bash
# Install all multimodal dependencies
pip install -r requirements_multimodal.txt

# Or install individually:
pip install vosk openai-whisper pyttsx3 pytesseract Pillow paddlepaddle paddleocr librosa soundfile
```

### System Dependencies

#### Windows
```bash
# Using Chocolatey
choco install tesseract espeak

# Or download manually:
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# eSpeak: http://espeak.sourceforge.net/download.html
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr espeak-ng
```

#### macOS
```bash
brew install tesseract espeak
```

## 📁 File Structure

After downloading, your directory structure will look like:
```
ai_backend/
├── models/
│   ├── phi-2-q4_k_m.gguf                    # LLM models
│   ├── llama-3.2-1b-instruct-q4_k_m.gguf
│   ├── vosk-model-small-en-us-0.15/         # STT models
│   └── ...
├── user_uploaded_files/                     # User multimodal files
│   └── {user_id}/
│       ├── audio_*.wav
│       ├── image_*.jpg
│       └── ...
└── requirements_multimodal.txt              # Dependencies
```

## 🔧 Troubleshooting

### Common Issues

1. **Unicode errors on Windows**
   - The script automatically handles Windows console encoding
   - If issues persist, use Windows Terminal or PowerShell

2. **Download failures**
   - Check internet connection
   - Use `--force` to retry failed downloads
   - Some models are large (1-7GB) and may take time

3. **System dependencies not found**
   - Install Tesseract and eSpeak system packages
   - Add them to your system PATH
   - Restart terminal after installation

4. **Permission errors**
   - Ensure write permissions to `models/` directory
   - Run as administrator if needed (Windows)

### Verification

Test your installation:
```bash
# Test LLM models
python -c "from app.modules.llm.local_model_manager import get_available_models; print(get_available_models())"

# Test multimodal imports
python -c "import vosk, whisper, pyttsx3, pytesseract, librosa; print('All imports successful')"
```

## 📚 Usage in Application

Once models are downloaded, they're automatically available in the application:

### LLM Models
```bash
# Query with specific local model
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "local_llm_model": "phi2"}'
```

### Multimodal APIs
```bash
# Speech-to-Text
curl -X POST "http://localhost:8000/api/audio/stt" \
  -F "file=@audio.wav" -F "provider=vosk"

# Text-to-Speech
curl -X POST "http://localhost:8000/api/audio/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "provider": "pyttsx3"}'

# OCR
curl -X POST "http://localhost:8000/api/vision/ocr" \
  -F "file=@document.jpg" -F "provider=tesseract"
```

## 🆘 Support

- Check the main README.md for general setup
- Review API_DOCUMENTATION.md for API usage
- Check logs in `logs/` directory for errors
- Ensure all dependencies are installed correctly

---

**Last Updated**: 2025-01-11  
**Compatible with**: Python 3.8+, Windows/Linux/macOS