"""Emotion detection providers."""

import logging
from .interfaces import IEmotionProvider, ProcessingResult

logger = logging.getLogger(__name__)


class BasicEmotionProvider(IEmotionProvider):
    """Basic emotion detection using audio features."""
    
    async def detect_emotion(self, audio_file_path: str) -> ProcessingResult:
        """Detect emotion from audio using basic heuristics."""
        try:
            import librosa
            import numpy as np
            
            # Load audio
            y, sr = librosa.load(audio_file_path)
            
            # Extract basic features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Simple heuristic classification
            mean_mfcc = np.mean(mfccs)
            mean_spectral = np.mean(spectral_centroids)
            mean_zcr = np.mean(zero_crossing_rate)
            
            # Basic emotion classification (simplified)
            if mean_spectral > 2000 and mean_zcr > 0.1:
                emotion = "excited"
                confidence = 0.7
            elif mean_spectral < 1000:
                emotion = "calm"
                confidence = 0.6
            elif mean_mfcc > 0:
                emotion = "positive"
                confidence = 0.5
            else:
                emotion = "neutral"
                confidence = 0.8
            
            return ProcessingResult(
                success=True,
                data={
                    "emotion": emotion,
                    "confidence": confidence,
                    "provider": "basic",
                    "features": {
                        "mean_mfcc": float(mean_mfcc),
                        "mean_spectral": float(mean_spectral),
                        "mean_zcr": float(mean_zcr)
                    }
                }
            )
            
        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="librosa not installed. Run: pip install librosa"
            )
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def get_provider_name(self) -> str:
        return "basic"


# Factory function
def create_emotion_provider(provider_name: str = "basic") -> IEmotionProvider:
    """Create emotion provider."""
    providers = {
        "basic": BasicEmotionProvider
    }
    
    provider_class = providers.get(provider_name, BasicEmotionProvider)
    return provider_class()