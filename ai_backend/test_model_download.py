#!/usr/bin/env python3
"""
Test model download and detection.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def test_model_system():
    """Test the model download and detection system."""
    
    print("=== MODEL SYSTEM TEST ===\n")
    
    # Test 1: List configured models
    print("1. CONFIGURED MODELS:")
    code, stdout, stderr = run_command("python scripts/download_hf_model.py --list")
    if code == 0:
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    # Test 2: Scan existing models
    print("\n2. EXISTING MODELS SCAN:")
    code, stdout, stderr = run_command("python scripts/download_hf_model.py --scan")
    if code == 0:
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    # Test 3: Try to download phi2 (with corrected URL)
    print("\n3. PHI-2 DOWNLOAD TEST:")
    print("Attempting to download Phi-2 model...")
    code, stdout, stderr = run_command("python scripts/download_hf_model.py --download phi2")
    if code == 0:
        print("✅ Phi-2 download successful")
        print(stdout)
    else:
        print(f"❌ Phi-2 download failed: {stderr}")
        print("Stdout:", stdout)
    
    # Test 4: Check models directory
    print("\n4. MODELS DIRECTORY:")
    models_dir = Path("models")
    if models_dir.exists():
        gguf_files = list(models_dir.glob("*.gguf"))
        if gguf_files:
            print(f"Found {len(gguf_files)} GGUF files:")
            for file_path in gguf_files:
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"  {file_path.name:40} | {size_mb:.1f} MB")
        else:
            print("No GGUF files found in models directory")
    else:
        print("Models directory does not exist")
    
    # Test 5: Test model API
    print("\n5. MODEL API TEST:")
    try:
        import requests
        response = requests.get("http://192.168.1.2:8000/api/models/list")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API working: {data['available_count']}/{data['total_count']} models available")
            print(f"Default model: {data['default_model']}")
        else:
            print(f"❌ API error: {response.status_code}")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    test_model_system()