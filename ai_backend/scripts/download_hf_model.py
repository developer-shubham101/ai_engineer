#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal AI model downloader.
Downloads LLM models (GGUF), STT models (Vosk), and other multimodal AI models.
"""

import argparse
import json
import sys
import zipfile
import os
from pathlib import Path

import requests

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


def get_config_path(config_type="llm"):
    """Get path to model configs."""
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "app" / "modules" / "config"
    
    if config_type == "llm":
        return config_dir / "local_models.json"
    elif config_type == "multimodal":
        return config_dir / "multimodal_models.json"
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def get_models_dir():
    """Get models directory path."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "models"


def load_model_config(config_type="llm"):
    """Load model configuration."""
    config_path = get_config_path(config_type)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {config_type} config: {e}")
        return None


def download_file(url: str, output_path: Path, resume: bool = True, extract_zip: bool = False):
    """Download file with resume support and optional zip extraction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For zip files, check if extracted directory exists
    if extract_zip and url.endswith('.zip'):
        extracted_dir = output_path.parent / output_path.stem
        if extracted_dir.exists() and not resume:
            print(f"Extracted directory {extracted_dir.name} already exists. Use --force to re-download.")
            return True

    # Check if file already exists
    if output_path.exists() and not resume:
        print(f"File {output_path.name} already exists. Use --force to overwrite.")
        return True

    # Get existing file size for resume
    existing_size = output_path.stat().st_size if output_path.exists() else 0

    headers = {}
    if resume and existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
        print(f"Resuming download from {existing_size} bytes...")

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Get total file size
        if 'content-length' in response.headers:
            total_size = int(response.headers['content-length'])
            if resume and existing_size > 0:
                total_size += existing_size
        else:
            total_size = None

        # Open file in appropriate mode
        mode = 'ab' if (resume and existing_size > 0 and response.status_code == 206) else 'wb'
        downloaded = existing_size if mode == 'ab' else 0

        print(f"Downloading {output_path.name}...")
        if total_size:
            print(f"Total size: {total_size / (1024 * 1024):.1f} MB")

        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\r{downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)",
                            end='', flush=True)
                    else:
                        print(f"\r{downloaded / (1024 * 1024):.1f} MB downloaded", end='', flush=True)

        print(f"\n[SUCCESS] Downloaded: {output_path.name}")
        
        # Extract zip if requested
        if extract_zip and url.endswith('.zip'):
            print(f"📦 Extracting {output_path.name}...")
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path.parent)
                print(f"✅ Extracted to: {output_path.parent}")
                # Optionally remove zip file after extraction
                # output_path.unlink()
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                return False
        
        return True

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False


def scan_existing_models():
    """Scan models directory for existing GGUF files."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []

    existing_files = []
    for file_path in models_dir.glob("*.gguf"):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        existing_files.append({
            "filename": file_path.name,
            "size_mb": size_mb,
            "path": str(file_path)
        })

    return existing_files


def list_available_models(model_type="all"):
    """List available models from configs."""
    if model_type in ["all", "llm"]:
        print("\n[LLM] LLM Models (GGUF):")
        print("=" * 80)
        list_llm_models()
    
    if model_type in ["all", "multimodal"]:
        print("\n[MULTIMODAL] Multimodal Models:")
        print("=" * 80)
        list_multimodal_models()


def list_llm_models():
    """List LLM models."""
    config = load_model_config("llm")
    if not config:
        return

    models_dir = get_models_dir()

    for key, info in config.get('models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        gguf_file = info.get('gguf_file', '')
        recommended = "[RECOMMENDED]" if info.get('recommended') else ""

        # Check if already downloaded
        if gguf_file:
            model_path = models_dir / gguf_file
            status = "[DOWNLOADED]" if model_path.exists() else "[AVAILABLE]"
            size_info = f" ({model_path.stat().st_size / (1024 * 1024):.1f} MB)" if model_path.exists() else ""
        else:
            status = "[NO URL]"
            size_info = ""

        print(f"{key:15} | {name:25} | {size:6} | {status}{size_info} {recommended}")
        if desc:
            print(f"                | {desc}")
        print()


def list_multimodal_models():
    """List multimodal models."""
    config = load_model_config("multimodal")
    if not config:
        return

    models_dir = get_models_dir()
    
    # STT Models
    print("[STT] Speech-to-Text Models:")
    print("-" * 60)
    for key, info in config.get('stt_models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        model_path = info.get('model_path', '')
        recommended = "[RECOMMENDED]" if info.get('recommended') else ""
        
        if model_path:
            full_path = Path(model_path)
            status = "[DOWNLOADED]" if full_path.exists() else "[AVAILABLE]"
        else:
            status = "[PACKAGE]"
        
        print(f"{key:15} | {name:20} | {size:8} | {status} {recommended}")
        if desc:
            print(f"                | {desc}")
        print()
    
    # TTS Models
    print("[TTS] Text-to-Speech Models:")
    print("-" * 60)
    for key, info in config.get('tts_models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        recommended = "[RECOMMENDED]" if info.get('recommended') else ""
        
        print(f"{key:15} | {name:20} | {size:8} | [PACKAGE] {recommended}")
        if desc:
            print(f"                | {desc}")
        print()
    
    # Vision Models
    print("[VISION] Vision/OCR Models:")
    print("-" * 60)
    for key, info in config.get('vision_models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        recommended = "[RECOMMENDED]" if info.get('recommended') else ""
        
        print(f"{key:15} | {name:20} | {size:8} | [PACKAGE] {recommended}")
        if desc:
            print(f"                | {desc}")
        print()
    
    # Whisper Models
    print("[WHISPER] Whisper Models:")
    print("-" * 60)
    for key, info in config.get('whisper_models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        recommended = "[RECOMMENDED]" if info.get('recommended') else ""
        
        print(f"{key:15} | {name:20} | {size:8} | [AUTO-DL] {recommended}")
        if desc:
            print(f"                | {desc}")
        print()
    
    # AI Vision Models from config
    print("[AI-VISION] AI Vision Models:")
    print("-" * 60)
    
    # Get vision models from LLM config
    llm_config = load_model_config("llm")
    if llm_config:
        for key, info in llm_config.get('models', {}).items():
            if info.get('model_type') == 'vision':
                name = info.get('name', key)
                size = info.get('size', 'Unknown')
                desc = info.get('description', '')
                recommended = "[RECOMMENDED]" if info.get('recommended') else ""
                
                print(f"{key:15} | {name:20} | {size:8} | [AUTO-DL] {recommended}")
                print(f"                | {desc}")
                print()


def download_vision_model(model_key: str, force: bool = False):
    """Download vision models from config."""
    config = load_model_config("llm")
    if not config:
        return False
    
    models = config.get('models', {})
    if model_key not in models or models[model_key].get('model_type') != 'vision':
        return False
    
    model_info = models[model_key]
    
    try:
        if model_key == "clip":
            from transformers import CLIPProcessor, CLIPModel
            repo = model_info.get('hf_repo')
            print(f"📥 Downloading {model_info.get('name')}...")
            CLIPModel.from_pretrained(repo)
            CLIPProcessor.from_pretrained(repo)
            print("✅ CLIP model downloaded successfully")
            
        elif model_key == "yolo":
            from ultralytics import YOLO
            model_file = model_info.get('model_file')
            print(f"📥 Downloading {model_info.get('name')}...")
            YOLO(model_file)
            print("✅ YOLO model downloaded successfully")
            
        elif model_key == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            repo = model_info.get('hf_repo')
            print(f"📥 Downloading {model_info.get('name')}...")
            BlipProcessor.from_pretrained(repo)
            BlipForConditionalGeneration.from_pretrained(repo)
            print("✅ BLIP model downloaded successfully")
            
        else:
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies for {model_key}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {model_key}: {e}")
        return False


def download_model(model_key: str, force: bool = False, model_type: str = "auto"):
    """Download a specific model."""
    # Check if it's a vision model first
    if model_key in ["clip", "yolo", "blip"]:
        return download_vision_model(model_key, force)
    
    # Try to find model in LLM config first
    if model_type in ["auto", "llm"]:
        if download_llm_model(model_key, force):
            return True
    
    # Try multimodal models
    if model_type in ["auto", "multimodal"]:
        if download_multimodal_model(model_key, force):
            return True
    
    print(f"❌ Model '{model_key}' not found in any config.")
    return False


def download_llm_model(model_key: str, force: bool = False):
    """Download LLM model."""
    config = load_model_config("llm")
    if not config:
        return False

    models = config.get('models', {})
    if model_key not in models:
        return False

    model_info = models[model_key]
    download_url = model_info.get('download_url')
    gguf_file = model_info.get('gguf_file')

    if not download_url or not gguf_file:
        print(f"❌ LLM model '{model_key}' missing download URL or filename.")
        return False

    models_dir = get_models_dir()
    output_path = models_dir / gguf_file

    # Check if already exists
    if output_path.exists() and not force:
        print(f"✅ LLM model '{model_key}' already downloaded: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
        print("   Use --force to re-download.")
        return True

    print(f"📥 Downloading LLM {model_info.get('name', model_key)} ({model_info.get('size', 'Unknown')})...")
    print(f"   URL: {download_url}")
    print(f"   Output: {output_path}")

    return download_file(download_url, output_path, resume=not force)


def download_multimodal_model(model_key: str, force: bool = False):
    """Download multimodal model."""
    config = load_model_config("multimodal")
    if not config:
        return False

    # Check STT models
    stt_models = config.get('stt_models', {})
    if model_key in stt_models:
        model_info = stt_models[model_key]
        download_url = model_info.get('download_url')
        model_path = model_info.get('model_path')
        
        if not download_url:
            print(f"❌ STT model '{model_key}' has no download URL.")
            return False
        
        # For zip files, download and extract
        if download_url.endswith('.zip'):
            zip_filename = f"{model_key}.zip"
            models_dir = get_models_dir()
            zip_path = models_dir / zip_filename
            
            # Check if already extracted
            if model_path:
                extracted_path = Path(model_path)
                if extracted_path.exists() and not force:
                    print(f"✅ STT model '{model_key}' already downloaded: {extracted_path}")
                    return True
            
            print(f"📥 Downloading STT {model_info.get('name', model_key)} ({model_info.get('size', 'Unknown')})...")
            print(f"   URL: {download_url}")
            print(f"   Extract to: {model_path}")
            
            return download_file(download_url, zip_path, resume=not force, extract_zip=True)
    
    # Other multimodal models don't need downloading (they're packages)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face GGUF models for local inference")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--type", choices=["llm", "multimodal", "all"], default="all", help="Model type to list/download")
    parser.add_argument("--download", type=str, help="Download specific model by key (e.g., clip, yolo, blip, mistral-7b)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--scan", action="store_true", help="Scan models directory for existing files")
    parser.add_argument("--install-deps", action="store_true", help="Show installation commands for multimodal dependencies")

    args = parser.parse_args()

    if args.list:
        list_available_models(args.type)
    elif args.scan:
        existing_models = scan_existing_models()
        if existing_models:
            print("Existing GGUF models in directory:")
            print("-" * 50)
            for model in existing_models:
                print(f"{model['filename']:40} | {model['size_mb']:.1f} MB")
        else:
            print("No GGUF models found in models directory.")
    elif args.install_deps:
        show_installation_commands()
    elif args.download:
        success = download_model(args.download, force=args.force, model_type=args.type)
        sys.exit(0 if success else 1)
    elif args.all:
        download_all_models(args.type, args.force)
        sys.exit(0)
    else:
        parser.print_help()


def show_installation_commands():
    """Show installation commands for multimodal dependencies."""
    print("\n[INSTALL] Multimodal Dependencies Installation:")
    print("=" * 60)
    
    print("\n[STT] Speech-to-Text:")
    print("pip install vosk")
    print("pip install openai-whisper")
    
    print("\n[TTS] Text-to-Speech:")
    print("pip install pyttsx3")
    print("# System: apt install espeak-ng (Linux) or choco install espeak (Windows)")
    
    print("\n[VISION] Vision/OCR:")
    print("pip install pytesseract pillow")
    print("pip install paddlepaddle paddleocr")
    print("pip install ultralytics  # YOLO")
    print("# System: apt install tesseract-ocr (Linux) or choco install tesseract (Windows)")
    
    print("\n[AI-VISION] AI Vision Models:")
    print("pip install transformers torch  # CLIP, BLIP")
    print("pip install ultralytics  # YOLO")
    
    print("\n[AUDIO] Audio Processing:")
    print("pip install librosa soundfile")
    
    print("\n[ALL] All at once:")
    print("pip install -r requirements_multimodal.txt")
    
    print("\n[NOTE] Some models auto-download on first use (Whisper, PaddleOCR)")


def download_all_models(model_type: str, force: bool):
    """Download all models of specified type."""
    success_count = 0
    total_count = 0
    
    if model_type in ["all", "llm"]:
        config = load_model_config("llm")
        if config:
            models = config.get('models', {})
            total_count += len(models)
            for model_key in models.keys():
                if download_llm_model(model_key, force=force):
                    success_count += 1
    
    if model_type in ["all", "multimodal"]:
        config = load_model_config("multimodal")
        if config:
            stt_models = config.get('stt_models', {})
            total_count += len(stt_models)
            for model_key in stt_models.keys():
                if download_multimodal_model(model_key, force=force):
                    success_count += 1
    
    print(f"\n✅ Downloaded {success_count}/{total_count} models successfully.")
    if success_count < total_count:
        print("💡 Some models require manual installation (see --install-deps)")


if __name__ == "__main__":
    main()
