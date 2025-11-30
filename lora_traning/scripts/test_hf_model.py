#!/usr/bin/env python3
"""
Test the HuggingFace model directly without GGUF conversion.
"""

from pathlib import Path
import sys

def test_hf_model(model_path: str):
    """Test HuggingFace model directly."""
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Testing HuggingFace model: {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        print("Model loaded successfully!")
        
        # Test query
        test_question = "What is the company policy on leave?"
        prompt = f"Question: {test_question}\nAnswer:"
        
        print(f"Testing query: {test_question}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        
        print(f"Response: {answer}")
        
        if len(answer) > 10:
            print("SUCCESS: Model is working correctly!")
            return True
        else:
            print("WARNING: Model response is too short")
            return False
            
    except ImportError as e:
        print(f"ERROR: Required packages not installed: {e}")
        print("Install with: pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    model_path = "models/distilgpt2-company-tuned"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"ERROR: Model directory not found: {model_path}")
        sys.exit(1)
    
    success = test_hf_model(model_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()