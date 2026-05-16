"""Audio processing API routes."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.modules.multimodal import (
    create_stt_provider, create_tts_provider, create_emotion_provider,
    LocalFileManager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["Audio"])


class TTSRequest(BaseModel):
    text: str
    conversation_id: str
    provider: str = "pyttsx3"


class AudioResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    file_path: str = None
    error: str = None


@router.post("/stt", response_model=AudioResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    provider: str = "vosk",
    conversation_id: str = "default",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Convert speech to text."""
    try:
        # Validate file type (allow common audio formats)
        allowed_types = ['audio/', 'video/']  # video/ for some audio containers
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.webm']
        
        file_ext = file.filename.lower().split('.')[-1] if file.filename else ''
        
        if not any(file.content_type.startswith(t) for t in allowed_types) and f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_manager = LocalFileManager()
        file_content = await file.read()
        file_path = await file_manager.save_uploaded_file(
            file_content, current_user["user_id"], "audio", conversation_id
        )
        
        # Process with STT
        stt_provider = create_stt_provider(provider)
        result = await stt_provider.transcribe(file_path)

        logger.info("STT completed: success=%s provider=%s file_path=%s", result.success, provider, file_path)
        
        return AudioResponse(
            success=result.success,
            data=result.data,
            file_path=file_path or "",
            error=result.error or ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("STT processing failed: %s", e)
        error_msg = str(e)
        if "RIFF" in error_msg:
            error_msg = "Audio format error. Please upload WAV, MP3, or other common audio formats. Install pydub for better format support: pip install pydub"
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/tts", response_model=AudioResponse)
async def text_to_speech(
    request: TTSRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Convert text to speech."""
    try:
        # Generate output path
        file_manager = LocalFileManager()
        import time
        timestamp = int(time.time())
        output_filename = f"tts_{request.conversation_id}_{timestamp}.wav"
        output_path = f"user_uploaded_files/{current_user['user_id']}/{output_filename}"
        
        # Process with TTS
        tts_provider = create_tts_provider(request.provider)
        result = await tts_provider.synthesize(request.text, output_path)

        logger.info(
            "TTS completed: success=%s provider=%s output_path=%s",
            result.success,
            request.provider,
            result.file_path or output_path,
        )
        return AudioResponse(
            success=result.success,
            data=result.data,
            file_path=result.file_path or "",
            error=result.error or ""
        )
        
    except Exception as e:
        logger.exception("TTS processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotion", response_model=AudioResponse)
async def detect_emotion(
    file: UploadFile = File(...),
    provider: str = "basic",
    conversation_id: str = "default",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Detect emotion from audio."""
    try:
        # Validate file type (allow common audio formats)
        allowed_types = ['audio/', 'video/']  # video/ for some audio containers
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.webm']
        
        file_ext = file.filename.lower().split('.')[-1] if file.filename else ''
        
        if not any(file.content_type.startswith(t) for t in allowed_types) and f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_manager = LocalFileManager()
        file_content = await file.read()
        file_path = await file_manager.save_uploaded_file(
            file_content, current_user["user_id"], "audio", conversation_id
        )
        
        # Process with emotion detector
        emotion_provider = create_emotion_provider(provider)
        result = await emotion_provider.detect_emotion(file_path)
        
        return AudioResponse(
            success=result.success,
            data=result.data,
            file_path=file_path,
            error=result.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Emotion detection failed: %s", e)
        error_msg = str(e)
        if "RIFF" in error_msg:
            error_msg = "Audio format error. Please upload WAV, MP3, or other common audio formats. Install pydub for better format support: pip install pydub"
        raise HTTPException(status_code=500, detail=error_msg)
