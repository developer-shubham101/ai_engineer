#!/usr/bin/env python3
"""
Hugging Face model downloader for local LLM models.
Downloads GGUF models for local inference.
"""

import argparse
import json
import sys
from pathlib import Path
import requests
from urllib.parse import urlparse

def get_config_path():
    """Get path to local models config."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "app" / "config" / "local_models.json"

def get_models_dir():
    """Get models directory path."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "models"

def load_model_config():
    """Load model configuration."""
    config_path = get_config_path()
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def download_file(url: str, output_path: Path, resume: bool = True):
    """Download file with resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
            print(f"Total size: {total_size / (1024*1024):.1f} MB")
        
        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\r{downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({percent:.1f}%)", end='', flush=True)
                    else:
                        print(f"\r{downloaded/(1024*1024):.1f} MB downloaded", end='', flush=True)
        
        print(f"\n✅ Downloaded: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def scan_existing_models():
    """Scan models directory for existing GGUF files."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []
    
    existing_files = []
    for file_path in models_dir.glob("*.gguf"):
        size_mb = file_path.stat().st_size / (1024*1024)
        existing_files.append({
            "filename": file_path.name,
            "size_mb": size_mb,
            "path": str(file_path)
        })
    
    return existing_files

def list_available_models():
    """List available models from config."""
    config = load_model_config()
    if not config:
        return
    
    models_dir = get_models_dir()
    existing_models = scan_existing_models()
    
    print("Configured models:")
    print("-" * 80)
    
    for key, info in config.get('models', {}).items():
        name = info.get('name', key)
        size = info.get('size', 'Unknown')
        desc = info.get('description', '')
        gguf_file = info.get('gguf_file', '')
        
        # Check if already downloaded
        if gguf_file:
            model_path = models_dir / gguf_file
            status = "✅ Downloaded" if model_path.exists() else "⬇️  Available"
            size_info = f" ({model_path.stat().st_size / (1024*1024):.1f} MB)" if model_path.exists() else ""
        else:
            status = "❌ No download URL"
            size_info = ""
        
        print(f"{key:15} | {name:25} | {size:4} | {status}{size_info}")
        if desc:
            print(f"                | {desc}")
        print()
    
    # Show existing models not in config
    existing_models = scan_existing_models()
    config_files = {info.get('gguf_file') for info in config.get('models', {}).values()}
    
    unrecognized = [m for m in existing_models if m['filename'] not in config_files]
    if unrecognized:
        print("\nUnrecognized models in directory:")
        print("-" * 50)
        for model in unrecognized:
            print(f"{model['filename']:40} | {model['size_mb']:.1f} MB | ✅ Available")

def download_model(model_key: str, force: bool = False):
    """Download a specific model."""
    config = load_model_config()
    if not config:
        return False
    
    models = config.get('models', {})
    if model_key not in models:
        print(f"❌ Model '{model_key}' not found in config.")
        print("Available models:", list(models.keys()))
        return False
    
    model_info = models[model_key]
    download_url = model_info.get('download_url')
    gguf_file = model_info.get('gguf_file')
    
    if not download_url or not gguf_file:
        print(f"❌ Model '{model_key}' missing download URL or filename.")
        return False
    
    models_dir = get_models_dir()
    output_path = models_dir / gguf_file
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"✅ Model '{model_key}' already downloaded: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print("   Use --force to re-download.")
        return True
    
    print(f"📥 Downloading {model_info.get('name', model_key)} ({model_info.get('size', 'Unknown')})...")
    print(f"   URL: {download_url}")
    print(f"   Output: {output_path}")
    
    return download_file(download_url, output_path, resume=not force)

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face GGUF models for local inference")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", type=str, help="Download specific model by key")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--scan", action="store_true", help="Scan models directory for existing files")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.scan:
        existing_models = scan_existing_models()
        if existing_models:
            print("Existing models in directory:")
            print("-" * 50)
            for model in existing_models:
                print(f"{model['filename']:40} | {model['size_mb']:.1f} MB")
        else:
            print("No GGUF models found in models directory.")
    elif args.download:
        success = download_model(args.download, force=args.force)
        sys.exit(0 if success else 1)
    elif args.all:
        config = load_model_config()
        if config:
            models = config.get('models', {})
            success_count = 0
            for model_key in models.keys():
                if download_model(model_key, force=args.force):
                    success_count += 1
            print(f"\n✅ Downloaded {success_count}/{len(models)} models successfully.")
        sys.exit(0)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()