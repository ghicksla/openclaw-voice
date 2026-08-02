"""Server-owned voice turn detection.

The semantic gate is adapted from pipecat-ai/smart-turn (BSD-2-Clause,
Copyright Daily). See https://github.com/pipecat-ai/smart-turn.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .vad import VoiceActivityDetector

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SECS = FRAME_MS / 1000

USER_SPEECH_STARTED = "user_speech_started"
EOT_PENDING = "eot_pending"
TURN_COMMITTED = "turn_committed"

REASON_SEMANTIC = "semantic"
REASON_CEILING = "ceiling"
REASON_MANUAL = "manual"
REASON_TIMEOUT = "timeout"

_TRUTHY = {"1", "true", "yes", "on"}
_ENV_PREFIX = "OPENCLAW_TURN_"
_MAX_PATIENCE_SECS = 20.0
_DEFAULT_GATE = object()


class TurnState(str, Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    PENDING_EOT = "pending_eot"
    AGENT_RESPONDING = "agent_responding"


@dataclass
class TurnEvent:
    type: str
    reason: Optional[str] = None
    audio: Optional[np.ndarray] = None
    sample_rate: int = SAMPLE_RATE


@dataclass
class TurnConfig:
    """Safe, environment-overridable turn detector settings."""

    min_speech_frames: int = 4
    min_silence_secs: float = 1.8
    fallback_silence_secs: float = 2.4
    recheck_interval_secs: float = 2.0
    patience_ceiling_secs: float = 18.0
    max_turn_secs: float = 45.0
    semantic_enabled: bool = True
    semantic_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.min_speech_frames = max(1, int(self.min_speech_frames))
        self.min_silence_secs = max(0.2, float(self.min_silence_secs))
        self.fallback_silence_secs = max(self.min_silence_secs, float(self.fallback_silence_secs))
        self.recheck_interval_secs = max(0.2, float(self.recheck_interval_secs))
        self.patience_ceiling_secs = min(
            max(self.min_silence_secs, float(self.patience_ceiling_secs)),
            _MAX_PATIENCE_SECS,
        )
        self.max_turn_secs = min(max(5.0, float(self.max_turn_secs)), 120.0)
        self.semantic_threshold = min(max(float(self.semantic_threshold), 0.0), 1.0)
        if not isinstance(self.semantic_enabled, bool):
            self.semantic_enabled = str(self.semantic_enabled).strip().lower() in _TRUTHY

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "TurnConfig":
        values: dict = {}
        fields = {field.name: field for field in dataclasses.fields(cls)}
        for name, field in fields.items():
            raw = os.getenv(f"{_ENV_PREFIX}{name.upper()}")
            if raw is not None:
                parsed = cls._parse(field, raw)
                if parsed is not None:
                    values[name] = parsed
        for name, raw in (overrides or {}).items():
            field = fields.get(name)
            if field is None:
                logger.warning("Ignoring unknown turn config key: {}", name)
                continue
            parsed = cls._parse(field, raw)
            if parsed is not None:
                values[name] = parsed
        return cls(**values)

    @staticmethod
    def _parse(field: dataclasses.Field, raw):
        try:
            if field.type in (bool, "bool"):
                return raw if isinstance(raw, bool) else str(raw).strip().lower() in _TRUTHY
            if field.type in (int, "int"):
                return int(raw)
            return float(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid turn config value {}={!r}", field.name, raw)
            return None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_SMART_TURN_DIR = Path(__file__).resolve().parents[2] / "models" / "smart-turn"


class SmartTurnGate:
    """ONNX semantic end-of-turn classifier from pipecat-ai/smart-turn-v3."""

    WINDOW_SECS = 8

    def __init__(self, model_path: Optional[str] = None):
        self.available = False
        self._session = None
        self._feature_extractor = None
        try:
            path = Path(model_path) if model_path else self._find_model()
            if path is None:
                raise FileNotFoundError(
                    "Smart Turn model not found; run "
                    "`python scripts/download_models.py smart-turn`"
                )
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(path), sess_options=options)
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=self.WINDOW_SECS)
            self.available = True
            logger.info("Smart Turn semantic gate ready ({})", path.name)
        except Exception as exc:
            logger.warning("Smart Turn semantic gate unavailable: {}", exc)

    @staticmethod
    def _find_model() -> Optional[Path]:
        matches = sorted(DEFAULT_SMART_TURN_DIR.glob("*.onnx"))
        return matches[-1] if matches else None

    def predict(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[float]:
        if not self.available:
            return None
        try:
            samples = np.asarray(audio, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                from scipy.signal import resample_poly

                divisor = math.gcd(sample_rate, SAMPLE_RATE)
                samples = resample_poly(
                    samples, SAMPLE_RATE // divisor, sample_rate // divisor
                ).astype(np.float32)
            max_samples = self.WINDOW_SECS * SAMPLE_RATE
            samples = samples[-max_samples:]
            if len(samples) < max_samples:
                samples = np.pad(samples, (max_samples - len(samples), 0))
            inputs = self._feature_extractor(
                samples,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )
            output = self._session.run(
                None, {"input_features": inputs.input_features.astype(np.float32)}
            )
            return float(output[0][0].item())
        except Exception as exc:
            logger.error("Smart Turn inference failed: {}", exc)
            return None


class TurnEngine:
    """Per-WebSocket state machine fed with a continuous mono float32 stream."""

    _PREROLL_SECS = 0.6
    _RESUME_SPEECH_FRAMES = 2

    def __init__(
        self,
        config: Optional[TurnConfig] = None,
        vad: Optional[VoiceActivityDetector] = None,
        gate=_DEFAULT_GATE,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.config = config or TurnConfig.from_env()
        self.vad = vad or VoiceActivityDetector()
        self.gate = (
            SmartTurnGate()
            if gate is _DEFAULT_GATE and self.config.semantic_enabled
            else (None if gate is _DEFAULT_GATE else gate)
        )
        if self.gate is not None and not getattr(self.gate, "available", True):
            self.gate = None
        self.semantic_active = bool(self.config.semantic_enabled and self.gate)
        self.sample_rate = sample_rate
        self.state = TurnState.IDLE
        self.last_semantic_prob: Optional[float] = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque()
        self._recent_speech: deque[bool] = deque(maxlen=max(8, self.config.min_speech_frames * 2))
        self._turn_frames: list[np.ndarray] = []
        self._silence_frames = 0
        self._pending_frames = 0
        self._resume_speech_frames = 0

    def feed(self, audio_chunk: np.ndarray) -> list[TurnEvent]:
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if self._pending.size:
            chunk = np.concatenate((self._pending, chunk))
        frame_size = max(1, int(self.sample_rate * FRAME_SECS))
        events: list[TurnEvent] = []
        offset = 0
        while offset + frame_size <= len(chunk):
            frame = chunk[offset : offset + frame_size].copy()
            offset += frame_size
            events.extend(self._process_frame(frame))
        self._pending = chunk[offset:].copy()
        return events

    def force_commit(self) -> list[TurnEvent]:
        if self.state in (TurnState.USER_SPEAKING, TurnState.PENDING_EOT):
            return [self._commit(REASON_MANUAL)]
        return []

    def set_agent_responding(self, active: bool) -> None:
        if active:
            self.state = TurnState.AGENT_RESPONDING
            self._clear_turn()
        elif self.state == TurnState.AGENT_RESPONDING:
            self.state = TurnState.IDLE

    def reset(self) -> None:
        self.state = TurnState.IDLE
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._recent_speech.clear()
        self._clear_turn()
        self.last_semantic_prob = None

    def diagnostics(self) -> dict:
        return {
            "state": self.state.value,
            "semantic_active": self.semantic_active,
            "last_semantic_prob": self.last_semantic_prob,
        }

    def _process_frame(self, frame: np.ndarray) -> list[TurnEvent]:
        is_speech = bool(self.vad.is_speech(frame, sample_rate=self.sample_rate))
        max_preroll = max(1, round(self._PREROLL_SECS / FRAME_SECS))
        self._preroll.append(frame)
        while len(self._preroll) > max_preroll:
            self._preroll.popleft()

        if self.state in (TurnState.IDLE, TurnState.AGENT_RESPONDING):
            if self.state == TurnState.AGENT_RESPONDING:
                return []
            self._recent_speech.append(is_speech)
            if sum(self._recent_speech) >= self.config.min_speech_frames:
                self.state = TurnState.USER_SPEAKING
                self._turn_frames = list(self._preroll)
                self._silence_frames = 0
                self._recent_speech.clear()
                return [TurnEvent(USER_SPEECH_STARTED)]
            return []

        self._turn_frames.append(frame)
        max_frames = max(1, round(self.config.max_turn_secs / FRAME_SECS))
        if len(self._turn_frames) >= max_frames:
            return [self._commit(REASON_CEILING)]

        if self.state == TurnState.USER_SPEAKING:
            if is_speech:
                self._silence_frames = 0
                return []
            self._silence_frames += 1
            silence = self._silence_frames * FRAME_SECS
            threshold = (
                self.config.min_silence_secs
                if self.semantic_active
                else self.config.fallback_silence_secs
            )
            if silence < threshold:
                return []
            if not self.semantic_active:
                return [self._commit(REASON_TIMEOUT)]
            self.state = TurnState.PENDING_EOT
            self._pending_frames = 0
            self._resume_speech_frames = 0
            events = [TurnEvent(EOT_PENDING)]
            events.extend(self._semantic_check())
            return events

        if is_speech:
            self._resume_speech_frames += 1
            if self._resume_speech_frames >= self._RESUME_SPEECH_FRAMES:
                self.state = TurnState.USER_SPEAKING
                self._silence_frames = 0
                self._pending_frames = 0
                self._resume_speech_frames = 0
            return []

        self._resume_speech_frames = 0
        self._silence_frames += 1
        self._pending_frames += 1
        silence = self._silence_frames * FRAME_SECS
        if silence >= self.config.patience_ceiling_secs:
            return [self._commit(REASON_CEILING)]
        recheck_frames = max(1, round(self.config.recheck_interval_secs / FRAME_SECS))
        if self._pending_frames >= recheck_frames:
            self._pending_frames = 0
            return self._semantic_check()
        return []

    def _semantic_check(self) -> list[TurnEvent]:
        probability = self.gate.predict(self._turn_audio(), self.sample_rate) if self.gate else None
        if probability is None:
            self.semantic_active = False
            self.state = TurnState.USER_SPEAKING
            if self._silence_frames * FRAME_SECS >= self.config.fallback_silence_secs:
                return [self._commit(REASON_TIMEOUT)]
            return []
        self.last_semantic_prob = probability
        if probability >= self.config.semantic_threshold:
            return [self._commit(REASON_SEMANTIC)]
        return []

    def _commit(self, reason: str) -> TurnEvent:
        audio = self._turn_audio()
        self.state = TurnState.IDLE
        self._clear_turn()
        self._recent_speech.clear()
        return TurnEvent(
            TURN_COMMITTED,
            reason=reason,
            audio=audio,
            sample_rate=self.sample_rate,
        )

    def _turn_audio(self) -> np.ndarray:
        return (
            np.concatenate(self._turn_frames)
            if self._turn_frames
            else np.zeros(0, dtype=np.float32)
        )

    def _clear_turn(self) -> None:
        self._turn_frames = []
        self._silence_frames = 0
        self._pending_frames = 0
        self._resume_speech_frames = 0
