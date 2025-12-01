#!/usr/bin/env python3
"""
Check if all required dependencies are installed and system is ready.
"""

import sys
import importlib
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "chromadb",
        "sentence_transformers",
        "numpy",
        "sqlite3",  # Built-in
        "PyJWT",
        "logging"  # Built-in
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == "PyJWT":
                importlib.import_module("jwt")
            else:
                importlib.import_module(package)
            print(f"OK {package}")
        except ImportError:
            missing.append(package)
            print(f"MISSING {package}")
    
    return missing

def check_directories():
    """Check if required directories exist."""
    base_dir = Path(__file__).parent
    required_dirs = [
        "database",
        "chroma_storage", 
        "models",
        "embeddings_models",
        "data"
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"OK {dir_name}/")
        else:
            missing_dirs.append(dir_name)
            print(f"MISSING {dir_name}/ (will be created)")
            dir_path.mkdir(exist_ok=True)
    
    return missing_dirs

def main():
    print("Checking AI Backend Setup")
    print("=" * 40)
    
    print("\nChecking Python packages:")
    missing_packages = check_requirements()
    
    print("\nChecking directories:")
    missing_dirs = check_directories()
    
    print("\nChecking modular architecture:")
    try:
        from app.modules.integration import get_container
        container = get_container()
        print("OK Modular architecture available")
    except Exception as e:
        print(f"ERROR Modular architecture error: {e}")
    
    print("\n" + "=" * 40)
    
    if missing_packages:
        print(f"ERROR Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("OK All requirements satisfied!")
        print("\nReady to run:")
        print("   python run_app.py")
        print("   OR")
        print("   uvicorn app.main:app --reload --port 8000")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)