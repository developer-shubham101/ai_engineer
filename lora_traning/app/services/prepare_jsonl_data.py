#!/usr/bin/env python3
"""
Load and prepare training data from agniholdings_train.jsonl.
Converts ChatML format to instruction-tuning format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_data(jsonl_path: Path) -> List[Dict[str, str]]:
    """Load training data from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of training pairs with 'instruction' and 'response' keys
    """
    training_pairs = []
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    logger.info(f"Loading training data from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Extract messages from ChatML format
                messages = data.get('messages', [])
                if len(messages) < 2:
                    logger.warning(f"Line {line_num}: Not enough messages, skipping")
                    continue
                
                # Find user and assistant messages
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    
                    if role == 'user':
                        user_msg = content
                    elif role == 'assistant':
                        assistant_msg = content
                
                if not user_msg or not assistant_msg:
                    logger.warning(f"Line {line_num}: Missing user or assistant message, skipping")
                    continue
                
                # Extract question from user message
                # Format: "Context:\n[context]\n\nQuestion: [question]"
                question = user_msg
                if "Question:" in user_msg:
                    question = user_msg.split("Question:")[-1].strip()
                
                # Create training pair
                training_pairs.append({
                    'instruction': question,
                    'response': assistant_msg,
                    'source': f'agniholdings_train.jsonl:L{line_num}'
                })
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Error processing - {e}")
                continue
    
    logger.info(f"Loaded {len(training_pairs)} training samples from JSONL")
    return training_pairs


def prepare_jsonl_training_data(jsonl_filename: str = "agniholdings_train.jsonl") -> List[Dict[str, str]]:
    """Prepare training data from JSONL file.
    
    Args:
        jsonl_filename: Name of the JSONL file in the data/ directory
        
    Returns:
        List of training pairs ready for model training
    """
    # Find the JSONL file in the data directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    jsonl_path = data_dir / jsonl_filename
    
    return load_jsonl_data(jsonl_path)


if __name__ == "__main__":
    # Test the data loader
    try:
        training_data = prepare_jsonl_training_data()
        
        print(f"\n{'='*60}")
        print(f"Successfully loaded {len(training_data)} training samples")
        print(f"{'='*60}\n")
        
        # Show first 3 samples
        for i, sample in enumerate(training_data[:3], 1):
            print(f"Sample {i}:")
            print(f"  Question: {sample['instruction'][:100]}...")
            print(f"  Answer: {sample['response'][:100]}...")
            print(f"  Source: {sample['source']}")
            print()
            
    except Exception as e:
        logger.exception(f"Failed to load training data: {e}")
