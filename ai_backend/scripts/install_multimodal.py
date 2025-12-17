#!/usr/bin/env python3
"""
Multimodal dependencies installer.
Installs Python packages and downloads models for multimodal AI features.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr}")
        return False


def install_python_packages():
    """Install Python packages for multimodal features."""
    packages = [
        # Speech-to-Text
        "vosk",
        "openai-whisper",
        
        # Text-to-Speech
        "pyttsx3",
        
        # Vision/OCR
        "pytesseract",
        "Pillow",
        "paddlepaddle",
        "paddleocr",
        
        # Audio Processing
        "librosa",
        "soundfile",
        
        # Utilities
        "python-multipart"
    ]
    
    print("\n📦 Installing Python packages...")
    print("=" * 50)
    
    success_count = 0
    for package in packages:
        if run_command([sys.executable, "-m", "pip", "install", package], 
                      f"Installing {package}"):
            success_count += 1
    
    print(f"\n✅ Installed {success_count}/{len(packages)} packages successfully")
    return success_count == len(packages)


def download_vosk_model():
    """Download Vosk STT model."""
    print("\n🎙️ Downloading Vosk STT model...")
    print("=" * 50)
    
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_hf_model.py"
    
    if not download_script.exists():
        print("❌ Download script not found")
        return False
    
    return run_command([sys.executable, str(download_script), "--download", "vosk-small-en", "--type", "multimodal"],
                      "Downloading Vosk small English model")


def check_system_dependencies():
    """Check for system dependencies."""
    print("\n🔍 Checking system dependencies...")
    print("=" * 50)
    
    dependencies = {
        "tesseract": "Tesseract OCR engine",
        "espeak": "eSpeak TTS engine (optional)"
    }
    
    available = []
    missing = []
    
    for cmd, desc in dependencies.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            print(f"✅ {desc} - Available")
            available.append(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {desc} - Not found")
            missing.append((cmd, desc))
    
    if missing:
        print(f"\n💡 To install missing system dependencies:")
        print("   Ubuntu/Debian: sudo apt install tesseract-ocr espeak-ng")
        print("   Windows: choco install tesseract espeak")
        print("   macOS: brew install tesseract espeak")
    
    return len(missing) == 0


def test_installations():
    """Test if installations work."""
    print("\n🧪 Testing installations...")
    print("=" * 50)
    
    tests = [
        ("import vosk", "Vosk STT"),
        ("import whisper", "Whisper STT"),
        ("import pyttsx3", "pyttsx3 TTS"),
        ("import pytesseract", "Tesseract OCR"),
        ("from PIL import Image", "Pillow"),
        ("import librosa", "librosa"),
        ("import soundfile", "soundfile")
    ]
    
    success_count = 0
    for test_cmd, desc in tests:
        try:
            subprocess.run([sys.executable, "-c", test_cmd], 
                          check=True, capture_output=True)
            print(f"✅ {desc} - Working")
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"❌ {desc} - Failed to import")
    
    print(f"\n✅ {success_count}/{len(tests)} packages working correctly")
    return success_count == len(tests)


def main():
    """Main installation process."""
    print("🚀 Multimodal AI Dependencies Installer")
    print("=" * 60)
    
    # Step 1: Install Python packages
    python_success = install_python_packages()
    
    # Step 2: Check system dependencies
    system_success = check_system_dependencies()
    
    # Step 3: Download models
    model_success = download_vosk_model()
    
    # Step 4: Test installations
    test_success = test_installations()
    
    # Summary
    print("\n📊 Installation Summary:")
    print("=" * 30)
    print(f"Python packages: {'✅' if python_success else '❌'}")
    print(f"System deps:     {'✅' if system_success else '⚠️'}")
    print(f"Models:          {'✅' if model_success else '⚠️'}")
    print(f"Tests:           {'✅' if test_success else '❌'}")
    
    if python_success and test_success:
        print("\n🎉 Multimodal AI setup complete!")
        print("   You can now use audio and vision processing features.")
        if not system_success:
            print("   Note: Some features may require system dependencies.")
    else:
        print("\n⚠️  Setup completed with issues.")
        print("   Check error messages above and install missing dependencies.")
    
    return python_success and test_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)