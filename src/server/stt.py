"""
Speech-to-Text module using Whisper.
"""

import asyncio

import numpy as np
from loguru import logger

try:
    from .audio_processing import preprocess as _preprocess_audio
except Exception as e:  # scipy/noisereduce missing — preprocessing is optional
    _preprocess_audio = None
    logger.warning(f"Audio preprocessing unavailable, STT will use raw audio: {e}")


class WhisperSTT:
    """Whisper-based Speech-to-Text."""
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self._backend = "mock"
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        # Prefer faster-whisper (CPU-friendly, no torch required).
        try:
            from faster_whisper import WhisperModel
            
            if self.device == "auto":
                self.device = "cpu"

            if self.device not in ("cpu", "cuda"):
                logger.warning(f"Unknown STT device '{self.device}', falling back to cpu")
                self.device = "cpu"

            compute_type = "float16" if self.device == "cuda" else "int8"
            
            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("✅ faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

        # Mock mode for testing
        logger.warning("⚠️ No STT backend - using mock mode")
        self._backend = "mock"
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
    
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "faster-whisper":
            if _preprocess_audio is not None:
                try:
                    audio = _preprocess_audio(audio)
                except Exception as e:
                    logger.warning(f"Audio preprocessing failed, using raw audio: {e}")
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        else:
            # Mock mode - return placeholder
            logger.debug(f"Mock STT: received {len(audio)} samples")
            return "[Mock transcription - install whisper for real STT]"
