"""File save tool for agent system."""
import os
from typing import Dict, Any


def save_text_file(filename: str, content: str) -> Dict[str, Any]:
    """Save text content to a file."""
    try:
        # Ensure safe file path
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'
        
        # Save to user_uploaded_files directory
        base_dir = "user_uploaded_files"
        os.makedirs(base_dir, exist_ok=True)
        
        filepath = os.path.join(base_dir, safe_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "filename": safe_filename,
            "filepath": filepath,
            "size": len(content),
            "status": "success"
        }
    except Exception as e:
        return {"filename": filename, "error": str(e), "status": "error"}