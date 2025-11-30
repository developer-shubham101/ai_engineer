from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import logging
from llama_cpp import Llama
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# Model cache
model_cache = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    use_llm: bool = False
    max_tokens: int = 256
    debug: bool = False
    local_llm_model: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    final_prompt: Optional[str] = None

def load_model(model_name: str):
    """Load model with caching (GGUF or HuggingFace)"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    models_dir = Path("models")
    
    # Try GGUF first
    gguf_path = models_dir / f"{model_name}.gguf"
    if gguf_path.exists():
        try:
            from llama_cpp import Llama
            model = Llama(
                model_path=str(gguf_path),
                n_ctx=512,
                n_threads=4,
                verbose=False
            )
            model_cache[model_name] = {"type": "gguf", "model": model}
            return model_cache[model_name]
        except Exception as e:
            logger.warning(f"Failed to load GGUF model {model_name}: {e}")
    
    # Try HuggingFace format
    hf_path = models_dir / model_name
    if hf_path.exists() and hf_path.is_dir():
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(str(hf_path))
            model = AutoModelForCausalLM.from_pretrained(str(hf_path))
            
            model_cache[model_name] = {
                "type": "hf", 
                "model": model, 
                "tokenizer": tokenizer
            }
            return model_cache[model_name]
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
    
    return None

@router.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest, req: Request):
    """Simple query endpoint for testing trained models"""
    
    if not request.use_llm:
        return QueryResponse(answer="LLM inference disabled")
    
    # Get model name
    model_name = request.local_llm_model
    if not model_name:
        # Try to get first available model from app state
        if hasattr(req.app.state, 'local_model_manager'):
            models = req.app.state.local_model_manager.get_available_models()
            if models:
                model_name = models[0]['name']
        
        if not model_name:
            raise HTTPException(status_code=404, detail="No models available")
    
    # Load model
    model = load_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Generate response
    try:
        prompt = f"Question: {request.question}\nAnswer:"
        
        if model["type"] == "gguf":
            # GGUF model
            response = model["model"](
                prompt,
                max_tokens=request.max_tokens,
                temperature=0.7,
                stop=["Question:", "\n\n"],
                echo=False
            )
            answer = response["choices"][0]["text"].strip()
            
        else:
            # HuggingFace model
            import torch
            tokenizer = model["tokenizer"]
            hf_model = model["model"]
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = hf_model.generate(
                    inputs.input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response[len(prompt):].strip()
        
        result = QueryResponse(answer=answer)
        if request.debug:
            result.final_prompt = prompt
        
        return result
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")