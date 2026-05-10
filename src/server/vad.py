"""
Voice Activity Detection module.
"""

import numpy as np
from loguru import logger


class VoiceActivityDetector:
    """Voice Activity Detection."""
    
    def __init__(self, aggressiveness: int = 2):
        # webrtcvad aggressiveness: 0 (least) .. 3 (most)
        self.aggressiveness = max(0, min(3, int(aggressiveness)))
        self.vad = None
        self._load_model()
    
    def _load_model(self):
        """Load VAD model (CPU-only, no torch)."""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.aggressiveness)
            logger.info("✅ WebRTC VAD loaded")
        except Exception as e:
            logger.warning(f"VAD not available: {e}")
            self.vad = None
    
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio contains speech."""
        if self.vad is None:
            return True  # Assume speech if no VAD
        try:
            if sample_rate not in (8000, 16000, 32000, 48000):
                return True

            # WebRTC VAD expects 16-bit mono PCM frames of 10/20/30ms.
            frame_ms = 30
            frame_len = int(sample_rate * frame_ms / 1000)
            if audio.size < frame_len:
                return True

            frame = audio[-frame_len:]
            pcm16 = (np.clip(frame, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm16, sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True
