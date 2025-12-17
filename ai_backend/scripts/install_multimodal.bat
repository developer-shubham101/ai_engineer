@echo off
echo 🚀 Multimodal AI Dependencies Installer (Windows)
echo ============================================================

echo.
echo 📦 Installing Python packages...
pip install vosk openai-whisper pyttsx3 pytesseract Pillow paddlepaddle paddleocr librosa soundfile python-multipart

echo.
echo 🎙️ Downloading Vosk model...
python download_hf_model.py --download vosk-small-en --type multimodal

echo.
echo 💡 System dependencies (optional):
echo    - Tesseract OCR: choco install tesseract
echo    - eSpeak TTS: choco install espeak
echo.
echo    Or download manually:
echo    - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
echo    - eSpeak: http://espeak.sourceforge.net/download.html

echo.
echo 🧪 Testing installation...
python -c "import vosk; print('✅ Vosk working')" 2>nul || echo "❌ Vosk failed"
python -c "import whisper; print('✅ Whisper working')" 2>nul || echo "❌ Whisper failed"
python -c "import pyttsx3; print('✅ pyttsx3 working')" 2>nul || echo "❌ pyttsx3 failed"
python -c "import pytesseract; print('✅ pytesseract working')" 2>nul || echo "❌ pytesseract failed"
python -c "import librosa; print('✅ librosa working')" 2>nul || echo "❌ librosa failed"

echo.
echo 🎉 Installation complete!
echo    Start the server with: python -m app.main
echo    Test multimodal APIs at: http://localhost:8000/docs

pause