"""Multimodal processing module."""

from .interfaces import (
    ISTTProvider, ITTSProvider, IVisionProvider, 
    IEmotionProvider, IFileManager, ProcessingResult
)
from .file_manager import LocalFileManager
from .stt_providers import create_stt_provider
from .tts_providers import create_tts_provider
from .vision_providers import create_vision_provider
from .emotion_providers import create_emotion_provider

__all__ = [
    "ISTTProvider", "ITTSProvider", "IVisionProvider", 
    "IEmotionProvider", "IFileManager", "ProcessingResult",
    "LocalFileManager",
    "create_stt_provider", "create_tts_provider", 
    "create_vision_provider", "create_emotion_provider"
]