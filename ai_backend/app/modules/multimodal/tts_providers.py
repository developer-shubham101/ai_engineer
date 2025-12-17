"""Text-to-Speech providers."""

import logging
import os
from .interfaces import ITTSProvider, ProcessingResult

logger = logging.getLogger(__name__)


class Pyttsx3TTSProvider(ITTSProvider):
    """pyttsx3 TTS provider (CPU-friendly)."""
    
    def __init__(self):
        self._engine = None
    
    async def synthesize(self, text: str, output_path: str) -> ProcessingResult:
        """Convert text to audio using pyttsx3."""
        try:
            import pyttsx3
            
            if not self._engine:
                self._engine = pyttsx3.init()
                # Set properties for better quality
                self._engine.setProperty('rate', 150)  # Speed
                self._engine.setProperty('volume', 0.9)  # Volume
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self._engine.save_to_file(text, output_path)
            self._engine.runAndWait()
            
            return ProcessingResult(
                success=True,
                data={
                    "text": text,
                    "provider": "pyttsx3",
                    "duration": len(text) * 0.1  # Rough estimate
                },
                file_path=output_path
            )
            
        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="pyttsx3 not installed. Run: pip install pyttsx3"
            )
        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def get_provider_name(self) -> str:
        return "pyttsx3"


class EspeakTTSProvider(ITTSProvider):
    """espeak TTS provider (lightweight)."""
    
    async def synthesize(self, text: str, output_path: str) -> ProcessingResult:
        """Convert text to audio using espeak."""
        try:
            import subprocess
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use espeak command
            cmd = [
                "espeak", 
                "-s", "150",  # Speed
                "-v", "en",   # Voice
                "-w", output_path,  # Output file
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return ProcessingResult(
                    success=True,
                    data={
                        "text": text,
                        "provider": "espeak",
                        "duration": len(text) * 0.1
                    },
                    file_path=output_path
                )
            else:
                return ProcessingResult(
                    success=False,
                    data={},
                    error=f"espeak error: {result.stderr}"
                )
                
        except FileNotFoundError:
            return ProcessingResult(
                success=False,
                data={},
                error="espeak not found. Install espeak-ng"
            )
        except Exception as e:
            logger.error(f"espeak TTS error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def get_provider_name(self) -> str:
        return "espeak"


# Factory function
def create_tts_provider(provider_name: str = "pyttsx3") -> ITTSProvider:
    """Create TTS provider."""
    providers = {
        "pyttsx3": Pyttsx3TTSProvider,
        "espeak": EspeakTTSProvider
    }
    
    provider_class = providers.get(provider_name, Pyttsx3TTSProvider)
    return provider_class()