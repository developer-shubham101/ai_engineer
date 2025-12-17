"""Multimodal interfaces for audio, vision, and media processing."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Generic result for multimodal processing."""
    success: bool
    data: Dict[str, Any]
    file_path: Optional[str] = None
    error: Optional[str] = None


class ISTTProvider(ABC):
    """Speech-to-Text provider interface."""
    
    @abstractmethod
    async def transcribe(self, audio_file_path: str) -> ProcessingResult:
        """Convert audio to text."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class ITTSProvider(ABC):
    """Text-to-Speech provider interface."""
    
    @abstractmethod
    async def synthesize(self, text: str, output_path: str) -> ProcessingResult:
        """Convert text to audio."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class IVisionProvider(ABC):
    """Vision processing provider interface."""
    
    @abstractmethod
    async def extract_text(self, image_path: str) -> ProcessingResult:
        """Extract text from image (OCR)."""
        pass
    
    @abstractmethod
    async def describe_image(self, image_path: str) -> ProcessingResult:
        """Generate image description."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class IEmotionProvider(ABC):
    """Emotion detection provider interface."""
    
    @abstractmethod
    async def detect_emotion(self, audio_file_path: str) -> ProcessingResult:
        """Detect emotion from audio."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class IFileManager(ABC):
    """File management interface."""
    
    @abstractmethod
    async def save_uploaded_file(self, file_content: bytes, user_id: str, 
                                file_type: str, conversation_id: str) -> str:
        """Save uploaded file and return path."""
        pass
    
    @abstractmethod
    async def get_file_path(self, user_id: str, filename: str) -> Optional[str]:
        """Get full file path."""
        pass
    
    @abstractmethod
    async def cleanup_old_files(self, user_id: str, days: int = 7) -> int:
        """Clean up old files."""
        pass