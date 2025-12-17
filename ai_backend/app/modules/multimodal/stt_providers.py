"""Speech-to-Text providers."""

import logging
import os
from .interfaces import ISTTProvider, ProcessingResult

logger = logging.getLogger(__name__)


class VoskSTTProvider(ISTTProvider):
    """Vosk STT provider (CPU-friendly)."""
    
    def __init__(self):
        self._model = None
    
    async def transcribe(self, audio_file_path: str) -> ProcessingResult:
        """Convert audio to text using Vosk."""
        try:
            # Lazy import to avoid dependency issues
            import vosk
            import json
            import wave
            
            if not self._model:
                # Download model if needed
                model_path = "models/vosk-model-small-en-us-0.15"
                if not os.path.exists(model_path):
                    return ProcessingResult(
                        success=False,
                        data={},
                        error="Vosk model not found. Please download vosk-model-small-en-us-0.15"
                    )
                self._model = vosk.Model(model_path)
            
            # Convert audio to WAV format if needed
            wav_file_path = await self._ensure_wav_format(audio_file_path)
            
            rec = vosk.KaldiRecognizer(self._model, 16000)
            
            with wave.open(wav_file_path, 'rb') as wf:
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        results.append(json.loads(rec.Result()))
                
                final_result = json.loads(rec.FinalResult())
                if final_result.get('text'):
                    results.append(final_result)
            
            text = ' '.join([r.get('text', '') for r in results]).strip()
            
            return ProcessingResult(
                success=True,
                data={
                    "text": text,
                    "provider": "vosk",
                    "confidence": 0.8  # Vosk doesn't provide confidence
                }
            )
            
        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="Vosk not installed. Run: pip install vosk"
            )
        except Exception as e:
            logger.error(f"Vosk STT error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    async def _ensure_wav_format(self, audio_file_path: str) -> str:
        """Convert audio file to WAV format if needed."""
        from .audio_utils import is_wav_file, convert_to_wav
        
        if is_wav_file(audio_file_path):
            return audio_file_path
        
        # Convert to WAV format
        return await convert_to_wav(audio_file_path, sample_rate=16000)
    
    def get_provider_name(self) -> str:
        return "vosk"


class WhisperSTTProvider(ISTTProvider):
    """Whisper STT provider."""
    
    def __init__(self):
        self._model = None
    
    async def transcribe(self, audio_file_path: str) -> ProcessingResult:
        """Convert audio to text using Whisper."""
        try:
            import whisper
            
            if not self._model:
                self._model = whisper.load_model("base")
            
            # Whisper handles most audio formats automatically
            # But we can add preprocessing if needed
            processed_path = await self._preprocess_audio(audio_file_path)
            
            result = self._model.transcribe(processed_path)
            
            return ProcessingResult(
                success=True,
                data={
                    "text": result["text"].strip(),
                    "provider": "whisper",
                    "language": result.get("language", "en")
                }
            )
            
        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="Whisper not installed. Run: pip install openai-whisper"
            )
        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    async def _preprocess_audio(self, audio_file_path: str) -> str:
        """Preprocess audio file if needed."""
        # Whisper is more flexible with audio formats
        # Just return the original path unless there are issues
        return audio_file_path
    
    def get_provider_name(self) -> str:
        return "whisper"


# Factory function
def create_stt_provider(provider_name: str = "vosk") -> ISTTProvider:
    """Create STT provider."""
    providers = {
        "vosk": VoskSTTProvider,
        "whisper": WhisperSTTProvider
    }
    
    provider_class = providers.get(provider_name, VoskSTTProvider)
    return provider_class()