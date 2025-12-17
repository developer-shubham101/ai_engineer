"""Audio processing utilities."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


async def convert_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """Convert audio file to WAV format."""
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    
    try:
        # Try pydub first (more formats supported)
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
        
        logger.info(f"Converted {input_path} to WAV using pydub")
        return output_path
        
    except ImportError:
        # Fallback to librosa + soundfile
        try:
            import librosa
            import soundfile as sf
            
            audio, sr = librosa.load(input_path, sr=sample_rate)
            sf.write(output_path, audio, sample_rate)
            
            logger.info(f"Converted {input_path} to WAV using librosa")
            return output_path
            
        except ImportError:
            raise Exception("Audio conversion failed. Install pydub or librosa: pip install pydub librosa soundfile")


def is_wav_file(file_path: str) -> bool:
    """Check if file is a valid WAV file."""
    try:
        import wave
        with wave.open(file_path, 'rb'):
            return True
    except:
        return False


def get_audio_info(file_path: str) -> dict:
    """Get audio file information."""
    try:
        import librosa
        
        # Get basic info without loading the full file
        duration = librosa.get_duration(path=file_path)
        
        # Try to get more detailed info
        try:
            y, sr = librosa.load(file_path, duration=1.0)  # Load just 1 second
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0],
                "format": Path(file_path).suffix.lower()
            }
        except:
            return {
                "duration": duration,
                "format": Path(file_path).suffix.lower()
            }
            
    except ImportError:
        # Basic info without librosa
        return {
            "format": Path(file_path).suffix.lower(),
            "size_mb": os.path.getsize(file_path) / (1024 * 1024)
        }
    except Exception as e:
        logger.warning(f"Could not get audio info for {file_path}: {e}")
        return {"format": Path(file_path).suffix.lower()}


def cleanup_temp_files(file_path: str):
    """Clean up temporary converted files."""
    if "_converted" in file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up {file_path}: {e}")