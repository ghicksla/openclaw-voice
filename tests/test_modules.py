"""
Unit tests for OpenClaw Voice modules.
"""

import pytest
import numpy as np
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.server.stt import WhisperSTT
from src.server.tts import ChatterboxTTS
from src.server.backend import AIBackend
from src.server.vad import VoiceActivityDetector
from src.server import audio_processing


class TestWhisperSTT:
    """Tests for Speech-to-Text module."""
    
    def test_init_loads_model(self):
        """Test that STT initializes (may be mock or real)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        assert stt is not None
        assert stt._backend in ["faster-whisper", "mock"]
    
    @pytest.mark.asyncio
    async def test_transcribe_returns_string(self):
        """Test that transcribe returns a string."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Create 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        result = await stt.transcribe(audio)
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_transcribe_with_noise(self):
        """Test transcription with random noise (should return something)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Random noise
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = await stt.transcribe(audio)
        assert isinstance(result, str)


class TestChatterboxTTS:
    """Tests for Text-to-Speech module."""
    
    def test_init_loads_model(self):
        """Test that TTS initializes (may be mock or real)."""
        tts = ChatterboxTTS()
        assert tts is not None
        assert tts._backend in ["elevenlabs", "chatterbox", "xtts", "pyttsx3", "mock"]
    
    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self):
        """Test that synthesize_aac returns encoded audio bytes or mock None."""
        tts = ChatterboxTTS()
        result = await tts.synthesize_aac("Hello world")
        assert result is None or isinstance(result, bytes)


class TestAIBackend:
    """Tests for AI Backend module."""
    
    def test_init_creates_client(self):
        """Test backend initialization."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
        )
        assert backend is not None
        assert backend.backend_type == "openai"
    
    def test_system_prompt_default(self):
        """Test default system prompt is set."""
        backend = AIBackend()
        assert backend.system_prompt is not None
        assert "voice assistant" in backend.system_prompt.lower()

    def test_max_tokens_default_and_override(self):
        """max_tokens defaults to 500 and is configurable."""
        assert AIBackend().max_tokens == 500
        assert AIBackend(max_tokens=222).max_tokens == 222
    
    def test_clear_history(self):
        """Test conversation history can be cleared."""
        backend = AIBackend()
        backend._history("test-user").append({"role": "user", "content": "test"})
        assert backend._history_by_user
        backend.clear_history("test-user")
        assert "test-user" not in backend._history_by_user
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_chat_returns_response(self):
        """Test actual API call (requires API key)."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        result = await backend.chat("Say 'test' and nothing else.")
        assert isinstance(result, str)
        assert len(result) > 0


class TestVAD:
    """Tests for Voice Activity Detection module."""
    
    def test_init(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector()
        assert vad is not None
    
    def test_is_speech_silence(self):
        """Test that silence is not detected as speech."""
        vad = VoiceActivityDetector()
        silence = np.zeros(16000, dtype=np.float32)
        # Should return True if no VAD model (assumes speech)
        # or False if VAD model is loaded and detects no speech
        result = vad.is_speech(silence)
        assert isinstance(result, bool)
    
    def test_is_speech_noise(self):
        """Test with random noise."""
        vad = VoiceActivityDetector()
        noise = np.random.randn(16000).astype(np.float32)
        result = vad.is_speech(noise)
        assert isinstance(result, bool)


class TestAudioProcessing:
    """Tests for the STT audio preprocessing pipeline."""

    def test_preprocess_preserves_dtype_and_shape_for_normal_audio(self):
        """A normal-length clip stays float32 and roughly the same length."""
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False).astype(np.float32)
        signal = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        out = audio_processing.preprocess(signal, sr)
        assert out.dtype == np.float32
        assert out.ndim == 1
        assert len(out) > 0

    def test_preprocess_short_audio_is_passthrough(self):
        """Clips under 250ms are returned untouched (no processing)."""
        short = np.zeros(1000, dtype=np.float32)
        out = audio_processing.preprocess(short)
        assert out is short

    def test_normalize_peaks_to_target(self):
        """Normalization scales the loudest sample to the target peak."""
        audio = (np.random.randn(16000).astype(np.float32)) * 0.05
        out = audio_processing.normalize(audio, target_peak=0.9)
        assert np.isclose(np.max(np.abs(out)), 0.9, atol=1e-3)

    def test_normalize_leaves_silence_alone(self):
        """Silence is not amplified (avoids boosting the noise floor)."""
        silence = np.zeros(16000, dtype=np.float32)
        out = audio_processing.normalize(silence)
        assert np.max(np.abs(out)) == 0.0


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_stt_tts_round_trip(self):
        """Test STT → TTS round trip (mock mode OK)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        tts = ChatterboxTTS()
        
        # Generate some audio (silence)
        input_audio = np.zeros(16000, dtype=np.float32)
        
        # Transcribe
        text = await stt.transcribe(input_audio)
        assert isinstance(text, str)
        
        # Synthesize (mock mode can return None when no TTS key is configured)
        if text.strip():
            output_audio = await tts.synthesize_aac(text)
        else:
            output_audio = await tts.synthesize_aac("Hello")
        
        assert output_audio is None or isinstance(output_audio, bytes)


# Run tests with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
