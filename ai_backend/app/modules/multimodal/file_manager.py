"""File management for multimodal uploads."""

import os
import time
from typing import Optional
from .interfaces import IFileManager


class LocalFileManager(IFileManager):
    """Local file system manager."""
    
    def __init__(self, base_path: str = "user_uploaded_files"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    async def save_uploaded_file(self, file_content: bytes, user_id: str, 
                                file_type: str, conversation_id: str) -> str:
        """Save uploaded file with generated name."""
        user_dir = os.path.join(self.base_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{file_type}_{conversation_id}_{timestamp}"
        
        # Add appropriate extension
        extensions = {
            "audio": ".wav", "tts": ".mp3", "image": ".jpg", 
            "doc": ".pdf", "ocr": ".txt"
        }
        filename += extensions.get(file_type, ".bin")
        
        file_path = os.path.join(user_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    async def get_file_path(self, user_id: str, filename: str) -> Optional[str]:
        """Get full file path."""
        file_path = os.path.join(self.base_path, user_id, filename)
        return file_path if os.path.exists(file_path) else None
    
    async def cleanup_old_files(self, user_id: str, days: int = 7) -> int:
        """Clean up old files."""
        user_dir = os.path.join(self.base_path, user_id)
        if not os.path.exists(user_dir):
            return 0
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned = 0
        
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            if os.path.getctime(file_path) < cutoff_time:
                os.remove(file_path)
                cleaned += 1
        
        return cleaned