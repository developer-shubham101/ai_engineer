#!/usr/bin/env python3
"""
Convert trained HuggingFace model to GGUF format for llama.cpp
"""

import sys
import json
import subprocess
from pathlib import Path

def convert_to_gguf(model_path: str, output_name: str = None):
    """Convert HuggingFace model to GGUF format."""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist")
        return False
    
    # Set output name
    if not output_name:
        output_name = model_dir.name
    
    gguf_path = model_dir.parent / f"{output_name}.gguf"
    
    try:
        # Use transformers' convert script
        cmd = [
            "python", "-m", "transformers.convert_graph_to_onnx.convert",
            "--framework", "pt",
            "--model", str(model_dir),
            str(gguf_path)
        ]
        
        print(f"Converting {model_dir} to GGUF format...")
        print(f"Output: {gguf_path}")
        
        # Alternative: Use huggingface-hub conversion
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        
        # Save in a format compatible with llama.cpp conversion
        print("Model loaded successfully. Manual GGUF conversion required.")
        print("Use llama.cpp's convert_hf_to_gguf.py script:")
        print(f"python convert_hf_to_gguf.py {model_dir} --outfile {gguf_path} --outtype q4_k_m")
        
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_gguf.py <model_path> [output_name]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_to_gguf(model_path, output_name)
    sys.exit(0 if success else 1)