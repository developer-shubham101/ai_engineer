#!/usr/bin/env python3
"""
Quick test to verify GGUF model can be loaded and used.
"""

from pathlib import Path
import sys

def test_gguf_model(model_path: str):
    """Test if GGUF model can be loaded."""
    
    try:
        from llama_cpp import Llama
        
        print(f"Testing GGUF model: {model_path}")
        
        # Load model
        model = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=2,
            verbose=False
        )
        
        print("Model loaded successfully!")
        
        # Test simple query
        test_question = "What is the company policy on leave?"
        prompt = f"Question: {test_question}\nAnswer:"
        
        print(f"Testing query: {test_question}")
        
        response = model(
            prompt,
            max_tokens=100,
            temperature=0.7,
            stop=["Question:", "\n\n"],
            echo=False
        )
        
        answer = response["choices"][0]["text"].strip()
        print(f"Response: {answer}")
        
        if len(answer) > 10:
            print("SUCCESS: Model is working correctly!")
            return True
        else:
            print("WARNING: Model response is too short")
            return False
            
    except ImportError:
        print("ERROR: llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    model_path = "models/distilgpt2-company-tuned.gguf"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    success = test_gguf_model(model_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()