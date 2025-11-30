#!/usr/bin/env python3
"""
Simple GGUF conversion fallback script.
Uses local llama.cpp installation or downloads minimal converter.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_local_converter():
    """Find local llama.cpp converter."""
    possible_paths = [
        "llama.cpp/convert_hf_to_gguf.py",
        "../llama.cpp/convert_hf_to_gguf.py",
        "convert_hf_to_gguf.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def convert_with_transformers(model_path: Path, output_path: Path):
    """Fallback: Use transformers to save in compatible format."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
        
        # Save in a format that can be manually converted
        temp_dir = model_path.parent / f"{model_path.name}_temp"
        temp_dir.mkdir(exist_ok=True)
        
        model.save_pretrained(str(temp_dir))
        tokenizer.save_pretrained(str(temp_dir))
        
        print(f"Model prepared for manual conversion at: {temp_dir}")
        print("To convert to GGUF manually:")
        print(f"1. Download llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
        print(f"2. Run: python llama.cpp/convert_hf_to_gguf.py {temp_dir} --outfile {output_path} --outtype q8_0")
        
        return True
        
    except Exception as e:
        print(f"Fallback conversion failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_gguf_convert_fixed.py <model_path> [output_name]")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else f"{model_path.name}.gguf"
    
    if not output_name.endswith('.gguf'):
        output_name += '.gguf'
    
    output_path = model_path.parent / output_name
    
    # Try to find local converter first
    converter = find_local_converter()
    
    if converter:
        print(f"Using local converter: {converter}")
        try:
            cmd = [
                sys.executable, converter,
                str(model_path),
                "--outfile", str(output_path),
                "--outtype", "q8_0"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("GGUF conversion successful!")
                print(f"Output: {output_path}")
                return
            else:
                print(f"Conversion failed: {result.stderr}")
        
        except Exception as e:
            print(f"Local conversion failed: {e}")
    
    # Fallback to transformers method
    print("Trying fallback method...")
    convert_with_transformers(model_path, output_path)

if __name__ == "__main__":
    main()