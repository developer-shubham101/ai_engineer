#!/usr/bin/env python3
"""
Improved GGUF conversion script for trained models.
Downloads llama.cpp and converts HuggingFace models to GGUF format.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile
import requests
import zipfile

def download_llama_cpp():
    """Download llama.cpp repository for conversion."""
    temp_dir = Path(tempfile.mkdtemp())
    llama_cpp_dir = temp_dir / "llama.cpp"
    
    print("Downloading llama.cpp...")
    
    try:
        # Clone llama.cpp repository
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git", 
            str(llama_cpp_dir)
        ], check=True, capture_output=True)
        
        print(f"Downloaded llama.cpp to {llama_cpp_dir}")
        return llama_cpp_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to download llama.cpp: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def convert_model_to_gguf(model_path: Path, output_path: Path, quantization: str = "q8_0"):
    """Convert HuggingFace model to GGUF format."""
    
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return False
    
    # Download llama.cpp
    llama_cpp_dir = download_llama_cpp()
    if not llama_cpp_dir:
        return False
    
    try:
        # Find the conversion script
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        
        if not convert_script.exists():
            print(f"Conversion script not found: {convert_script}")
            return False
        
        print(f"Converting {model_path} to GGUF format...")
        print(f"Output: {output_path}")
        print(f"Quantization: {quantization}")
        
        # Run conversion
        cmd = [
            sys.executable, str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", quantization
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("GGUF conversion successful!")
            return True
        else:
            print(f"Conversion failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Conversion error: {e}")
        return False
    finally:
        # Cleanup - handle Windows permission issues
        try:
            if llama_cpp_dir.exists():
                # Force remove read-only files on Windows
                def handle_remove_readonly(func, path, exc):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                shutil.rmtree(llama_cpp_dir.parent, onerror=handle_remove_readonly)
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed: {cleanup_error}")

def main():
    """Main conversion function."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_gguf_improved.py <model_path> [output_name] [quantization]")
        print("Example: python convert_to_gguf_improved.py models/distilgpt2-company-tuned")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else model_path.name
    quantization = sys.argv[3] if len(sys.argv) > 3 else "q8_0"
    
    # Ensure output has .gguf extension
    if not output_name.endswith('.gguf'):
        output_name += '.gguf'
    
    output_path = model_path.parent / output_name
    
    success = convert_model_to_gguf(model_path, output_path, quantization)
    
    if success:
        print(f"\nConversion completed!")
        print(f"GGUF file: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("\nConversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()