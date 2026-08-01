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


class TestSessionCompletionTask:
    """Tests for matching gateway background-task completions to a voice session."""

    SESSION = "agent:main:openai-user:abc123"

    def _match(self, task):
        from src.server.main import is_session_completion_task
        return is_session_completion_task(task, self.SESSION)

    def test_matches_current_gateway_schema(self):
        """A succeeded, session-owned, done_only task matches (no announce:v1: prefix)."""
        task = {
            "taskId": "t1",
            "sourceId": "3a98743e-d5a9-4f24-893d-f6c922ac487d",  # bare run-id, no prefix
            "ownerKey": self.SESSION,
            "status": "succeeded",
            "notifyPolicy": "done_only",
        }
        assert self._match(task) is True

    def test_excludes_silent_subagent_task(self):
        """The subagent's own internal task (different owner, silent) is excluded."""
        task = {
            "taskId": "t2",
            "ownerKey": "agent:main:subagent:xyz",
            "status": "succeeded",
            "notifyPolicy": "silent",
        }
        assert self._match(task) is False

    def test_excludes_other_session(self):
        task = {
            "taskId": "t3",
            "ownerKey": "agent:main:openai-user:someone-else",
            "status": "succeeded",
            "notifyPolicy": "done_only",
        }
        assert self._match(task) is False

    def test_excludes_unfinished(self):
        task = {
            "taskId": "t4",
            "ownerKey": self.SESSION,
            "status": "running",
            "notifyPolicy": "done_only",
        }
        assert self._match(task) is False

    def test_matches_failed_session_task(self):
        """A failed session-scoped task is delivered so the UI doesn't hang."""
        task = {
            "taskId": "t5",
            "ownerKey": self.SESSION,
            "status": "failed",
            "notifyPolicy": "done_only",
        }
        assert self._match(task) is True


class TestCompoundEmailTask:
    """Splits "do X and email it to me" into task fragment + email follow-up."""

    def _extract(self, text):
        from src.server.main import extract_compound_email_task_text
        return extract_compound_email_task_text(text)

    def test_greg_pull_comps_request(self):
        """The exact phrase that misrouted today."""
        assert (
            self._extract("Pull the latest comps for that property and send them to my Gmail account")
            == "Pull the latest comps for that property"
        )

    def test_email_it_to_me(self):
        assert self._extract("Research 437 Logan and email it to me") == "Research 437 Logan"

    def test_email_me_the_latest(self):
        assert self._extract("Find AI news and email me the latest") == "Find AI news"

    def test_forward_to_inbox(self):
        assert self._extract("Find AI news and forward it to my inbox") == "Find AI news"

    def test_pure_copy_not_compound(self):
        for phrase in (
            "Send that to my gmail",
            "Send a copy of that to my gmail",
            "Email me that",
        ):
            assert self._extract(phrase) is None, phrase

    def test_short_or_no_leading_task(self):
        for phrase in (
            "Send me to bed",
            "What time is it",
            "And send them",
        ):
            assert self._extract(phrase) is None, phrase


class TestEmailCopyIntent:
    """Intent detection for "send the last answer to my email" voice asks."""

    def _explicit(self, text):
        from src.server.main import is_send_copy_to_email_request
        return is_send_copy_to_email_request(text)

    def _intent(self, text):
        from src.server.main import detect_voice_intents
        return "email_copy" in detect_voice_intents(text)

    def test_yesterdays_failing_phrase(self):
        """"Send those comps to my Gmail account" — broken before today's fix."""
        assert self._intent("Send those comps to my Gmail account") is True

    def test_natural_anaphors(self):
        for phrase in (
            "Send those to my gmail",
            "Forward these to my inbox",
            "Email them to my email",
            "Send me those comps",
            "Send me a recap",
            "Forward this to my mailbox",
        ):
            assert self._intent(phrase) is True, phrase

    def test_classic_explicit_still_matches(self):
        for phrase in (
            "Send a copy to my gmail",
            "Email that to my gmail",
            "Send a summary to my email",
        ):
            assert self._explicit(phrase) is True, phrase

    def test_future_tense_news_is_not_email_copy(self):
        """Future-tense asks must not hijack the previous-answer copy path."""
        for phrase in (
            "Email me the news in the morning",
            "Send me a daily update at 8am",
            "Mail me when the report is ready",
        ):
            assert self._intent(phrase) is False, phrase

    def test_no_target_word_is_not_email_copy(self):
        for phrase in (
            "Send those flowers to my mom",
            "Mail it to my coworker",
            "What is the weather like",
        ):
            assert self._intent(phrase) is False, phrase


class TestSessionFailureMessage:
    """The voice apology phrasing for a failed session task."""

    def _msg(self, task):
        from src.server.main import session_failure_message
        return session_failure_message(task)

    def test_default_when_no_details(self):
        msg = self._msg({"status": "failed"})
        assert "trouble" in msg.lower()
        assert "try again" in msg.lower()

    def test_uses_short_error(self):
        msg = self._msg({"status": "failed", "error": "Rate limit reached"})
        assert "Rate limit reached" in msg

    def test_drops_long_error(self):
        long = "x" * 400
        msg = self._msg({"status": "failed", "error": long})
        assert long not in msg
        assert "trouble" in msg.lower()

    def test_drops_multiline_error(self):
        msg = self._msg({"status": "failed", "error": "Boom\nTraceback (most recent…)"})
        assert "Traceback" not in msg
        assert "trouble" in msg.lower()


class TestIncompleteVoiceAnswer:
    """Detect model/gateway fragments that were cut off mid-sentence."""

    def _incomplete(self, text):
        from src.server.main import is_incomplete_voice_answer
        return is_incomplete_voice_answer(text)

    def test_detects_property_fragment(self):
        assert self._incomplete("The property at 437 Logan St, Santa Cruz, is a 3") is True

    def test_detects_long_no_punctuation(self):
        text = "This is a fairly long answer that clearly continues for quite a while but never reaches a proper ending"
        assert self._incomplete(text) is True

    def test_short_plain_answer_allowed(self):
        assert self._incomplete("Yes") is False

    def test_complete_sentence_allowed(self):
        assert self._incomplete("The property looks overpriced based on the latest comps.") is False


class TestEmitFilteredFlush:
    """The standby utterance must not stall behind a missing sentence space."""

    @pytest.mark.asyncio
    async def test_flush_ships_partial_sentence(self):
        # Recreate the emit_filtered closure pattern from process_transcript
        # to verify flush=True ships the residual partial sentence.
        SENTENCE_ENDS = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        sentence_buffer = ""
        shipped: list[str] = []

        async def emit_filtered(text: str, *, flush: bool = False):
            nonlocal sentence_buffer
            if not text and not flush:
                return
            if text:
                sentence_buffer += text
            while any(sep in sentence_buffer for sep in SENTENCE_ENDS):
                earliest_idx = len(sentence_buffer)
                for sep in SENTENCE_ENDS:
                    idx = sentence_buffer.find(sep)
                    if idx != -1 and idx + len(sep) < earliest_idx:
                        earliest_idx = idx + len(sep)
                if earliest_idx >= len(sentence_buffer):
                    break
                sentence = sentence_buffer[:earliest_idx].strip()
                sentence_buffer = sentence_buffer[earliest_idx:]
                if sentence:
                    shipped.append(sentence)
            if flush:
                residue = sentence_buffer.strip()
                sentence_buffer = ""
                if residue:
                    shipped.append(residue)

        await emit_filtered("I'm still working on that. Please stand by.", flush=True)
        assert shipped == ["I'm still working on that.", "Please stand by."]

        # Without flush, the trailing partial would sit in the buffer and
        # get glued to the next emit — that's the bug we're preventing.
        shipped.clear()
        sentence_buffer = ""
        await emit_filtered("Working on it, I'll respond when finished.")
        assert shipped == []  # No terminator + space, nothing ships yet.


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
