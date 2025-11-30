# app/services/model_training_service.py
"""
Model training service for fine-tuning Llama 3.2 1B on company data.
Exports trained models to the models/ directory for use with RAG system.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import torch
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("Training dependencies not available. Install: pip install transformers datasets torch")

# Training data preparation - read from data/ directory and subdirectories
def read_training_data_from_files(data_dir: Path = Path("data"), max_samples: int = 1000) -> List[Dict[str, str]]:
    """Read training data directly from markdown files in data directory."""
    training_pairs = []
    
    # Read directly from data/company/*.md files
    company_dir = data_dir / "company"
    if not company_dir.exists():
        logger.warning(f"Company data directory not found: {company_dir}")
        return []
    
    logger.info(f"Reading company documents from {company_dir}")
    
    # Process all markdown files
    for file_path in company_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                continue
                
            document_name = file_path.stem.replace('_', ' ').title()
            
            # Create multiple Q&A pairs from each document
            questions = [
                f"What is {document_name} about?",
                f"Tell me about {document_name.lower()}.",
                f"Explain the {document_name.lower()} policy.",
                f"What should I know about {document_name.lower()}?"
            ]
            
            # Add document-level Q&A pairs
            for question in questions:
                training_pairs.append({
                    "instruction": question,
                    "response": content[:800],  # Limit response length
                    "source": document_name
                })
            
            # Parse sections for more specific Q&A
            sections = content.split('\n#')
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) < 50:
                    continue
                    
                if i > 0:  # Skip first section (already used above)
                    lines = section.split('\n')
                    if lines:
                        header = lines[0].lstrip('#').strip()
                        body = '\n'.join(lines[1:]).strip()
                        
                        if body and header:
                            training_pairs.append({
                                "instruction": f"What is the company policy on {header.lower()}?",
                                "response": body[:600],
                                "source": document_name
                            })
            
            if len(training_pairs) >= max_samples:
                break
                
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
    
    logger.info(f"Created {len(training_pairs)} training pairs from {company_dir}")
    return training_pairs


def prepare_training_data(max_samples: int = 1000) -> List[Dict[str, str]]:
    """Read training data from .md and .txt files in data/ directory."""
    if not TRAINING_AVAILABLE:
        raise RuntimeError("Training dependencies not installed")

    logger.info("Preparing training data from company documents")

    try:
        # root/data/company/*.md
        app_path = Path(__file__).parent.parent.parent
        data_path = app_path / "data"
        training_pairs = read_training_data_from_files(data_path, max_samples)

        if not training_pairs:
            raise ValueError("No .md or .txt files found in data/ directory")

        logger.info(f"Prepared {len(training_pairs)} training samples from company documents")
        return training_pairs

    except Exception as e:
        logger.exception("Failed to prepare training data: %s", e)
        raise


def format_training_data(samples: List[Dict[str, str]]) -> Dataset:
    """Format data for instruction tuning."""
    formatted_data = []

    for sample in samples:
        # Use simple instruction format for DistilGPT2
        text = f"Question: {sample['instruction']}\nAnswer: {sample['response']}"
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


class ModelTrainingService:
    """Service for training Llama 3.2 1B on company documents."""
    
    def __init__(self):
        # Use an open-source model that doesn't require authentication
        self.model_name = "distilgpt2"  # Lightweight, open-source alternative
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def train_model(
        self, 
        output_name: str = "distilgpt2-company-tuned",
        max_samples: int = 1000,
        epochs: int = 3,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """Train Llama 3.2 1B on company data."""
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training dependencies not installed")
            
        logger.info(f"Starting training of {self.model_name}")
        
        try:
            # Prepare data
            samples = prepare_training_data(max_samples)
            dataset = format_training_data(samples)
            
            # Load model and tokenizer
            logger.info("Loading model and tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Tokenize dataset
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,  # Don't pad here, let data collator handle it
                    max_length=512,
                    return_tensors=None  # Return lists, not tensors
                )
                # Add labels for language modeling
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            
            # Training arguments
            output_dir = self.models_dir / f"{output_name}-training"
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # For efficiency
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting training...")
            trainer.train()
            
            # Save final model
            final_model_dir = self.models_dir / output_name
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
            
            # Convert to GGUF format for llama.cpp
            gguf_path = self.convert_to_gguf(final_model_dir, output_name)
            
            # Save training info
            training_info = {
                "model_name": output_name,
                "base_model": self.model_name,
                "training_samples": len(samples),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "trained_at": datetime.utcnow().isoformat(),
                "model_path": str(final_model_dir),
                "gguf_path": str(gguf_path) if gguf_path else None
            }
            
            info_file = self.models_dir / f"{output_name}.json"
            with open(info_file, 'w') as f:
                json.dump(training_info, f, indent=2)
                
            logger.info(f"Training completed. Model saved to {final_model_dir}")
            return training_info
            
        except Exception as e:
            logger.exception("Training failed: %s", e)
            raise
    
    def convert_to_gguf(self, model_dir: Path, output_name: str) -> Optional[Path]:
        """Convert trained model to GGUF format for llama.cpp."""
        try:
            # This requires llama.cpp convert script
            gguf_path = self.models_dir / f"{output_name}.gguf"
            
            # Try to find convert script
            convert_script = None
            possible_paths = [
                "llama.cpp/convert_hf_to_gguf.py",
                "convert_hf_to_gguf.py",
                "/usr/local/bin/convert_hf_to_gguf.py"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    convert_script = path
                    break
                    
            if not convert_script:
                logger.warning("GGUF conversion script not found. Model saved in HuggingFace format only.")
                return None
                
            # Run conversion
            import subprocess
            cmd = [
                "python", convert_script,
                str(model_dir),
                "--outfile", str(gguf_path),
                "--outtype", "q4_k_m"  # 4-bit quantization
            ]
            
            logger.info("Converting to GGUF format...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"GGUF conversion successful: {gguf_path}")
                return gguf_path
            else:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.exception("GGUF conversion error: %s", e)
            return None
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models in the models directory."""
        trained_models = []
        
        for info_file in self.models_dir.glob("*.json"):
            try:
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    
                # Check if model files exist
                model_path = Path(model_info.get("model_path", ""))
                gguf_path = Path(model_info.get("gguf_path", "")) if model_info.get("gguf_path") else None
                
                model_info["hf_exists"] = model_path.exists()
                model_info["gguf_exists"] = gguf_path.exists() if gguf_path else False
                
                trained_models.append(model_info)
                
            except Exception as e:
                logger.warning(f"Failed to load model info from {info_file}: {e}")
                
        return trained_models


# Global service instance
_training_service = ModelTrainingService()


async def train_company_model(
    output_name: str = "distilgpt2-company-tuned",
    max_samples: int = 1000,
    epochs: int = 3,
    learning_rate: float = 2e-5
) -> Dict[str, Any]:
    """Train a model on company data."""
    return _training_service.train_model(
        output_name=output_name,
        max_samples=max_samples,
        epochs=epochs,
        learning_rate=learning_rate
    )


def get_trained_models() -> List[Dict[str, Any]]:
    """Get list of trained models."""
    return _training_service.list_trained_models()


def is_training_available() -> bool:
    """Check if training dependencies are available."""
    return TRAINING_AVAILABLE