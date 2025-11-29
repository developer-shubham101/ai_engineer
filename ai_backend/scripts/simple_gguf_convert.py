#!/usr/bin/env python3
"""
Simple GGUF conversion using available tools
"""

import os
import sys
import subprocess
from pathlib import Path

def download_and_convert():
    """Download llama.cpp and convert model to GGUF."""
    
    model_path = Path("models/distilgpt2-company-tuned")
    if not model_path.exists():
        print("Error: Trained model not found")
        return False
    
    # Check if we can use git to clone llama.cpp
    try:
        print("Cloning llama.cpp repository...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git", "temp_llama_cpp"
        ], check=True, cwd=".")
        
        # Use the conversion script
        convert_script = Path("temp_llama_cpp/convert_hf_to_gguf.py")
        if convert_script.exists():
            print("Converting model to GGUF...")
            subprocess.run([
                "python", str(convert_script),
                str(model_path),
                "--outfile", "models/distilgpt2-company-tuned.gguf",
                "--outtype", "q4_k_m"
            ], check=True)
            
            print("✅ Conversion successful!")
            print("GGUF file: models/distilgpt2-company-tuned.gguf")
            
            # Clean up
            import shutil
            shutil.rmtree("temp_llama_cpp")
            return True
        else:
            print("Conversion script not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = download_and_convert()
    if not success:
        print("\nAlternative: Manual conversion steps:")
        print("1. git clone https://github.com/ggerganov/llama.cpp.git")
        print("2. cd llama.cpp")
        print("3. python convert_hf_to_gguf.py ../models/distilgpt2-company-tuned --outfile ../models/distilgpt2-company-tuned.gguf --outtype q4_k_m")