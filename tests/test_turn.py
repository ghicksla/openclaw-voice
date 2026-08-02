"""Unit tests for the server-owned turn engine."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from src.server.turn import (
    EOT_PENDING,
    FRAME_SECS,
    REASON_CEILING,
    REASON_MANUAL,
    REASON_SEMANTIC,
    REASON_TIMEOUT,
    SmartTurnGate,
    TURN_COMMITTED,
    USER_SPEECH_STARTED,
    TurnConfig,
    TurnEngine,
    TurnState,
)


class ScriptedVAD:
    def __init__(self):
        self.values = deque()

    def push(self, *values):
        self.values.extend(values)

    def is_speech(self, _audio, sample_rate=16000):
        return self.values.popleft() if self.values else False


class FakeGate:
    available = True

    def __init__(self, *probabilities):
        self.probabilities = deque(probabilities or (0.9,))
        self.last = self.probabilities[-1]
        self.calls = 0

    def predict(self, _audio, _sample_rate=16000):
        self.calls += 1
        if self.probabilities:
            self.last = self.probabilities.popleft()
        return self.last


def make_engine(gate=None, **config):
    vad = ScriptedVAD()
    engine = TurnEngine(
        config=TurnConfig(**config),
        vad=vad,
        gate=gate,
        sample_rate=16000,
    )
    return engine, vad


def feed(engine, vad, values):
    events = []
    frame = np.full(int(16000 * FRAME_SECS), 0.1, dtype=np.float32)
    for value in values:
        vad.push(value)
        events.extend(engine.feed(frame if value else np.zeros_like(frame)))
    return events


def frames(seconds):
    return int(seconds / FRAME_SECS) + 2


def test_semantic_commit_after_silence():
    engine, vad = make_engine(FakeGate(0.9))
    events = feed(engine, vad, [True] * 6)
    assert [event.type for event in events] == [USER_SPEECH_STARTED]

    events = feed(engine, vad, [False] * frames(1.8))
    assert EOT_PENDING in [event.type for event in events]
    committed = next(event for event in events if event.type == TURN_COMMITTED)
    assert committed.reason == REASON_SEMANTIC
    assert committed.audio.size > 0
    assert engine.state == TurnState.IDLE


def test_incomplete_semantic_turn_waits_until_ceiling():
    engine, vad = make_engine(
        FakeGate(0.1),
        min_silence_secs=0.2,
        recheck_interval_secs=0.2,
        patience_ceiling_secs=0.8,
    )
    feed(engine, vad, [True] * 6)
    events = feed(engine, vad, [False] * frames(0.9))
    committed = [event for event in events if event.type == TURN_COMMITTED]
    assert len(committed) == 1
    assert committed[0].reason == REASON_CEILING


def test_resumed_speech_cancels_pending_eot():
    engine, vad = make_engine(FakeGate(0.1, 0.9), min_silence_secs=0.2)
    feed(engine, vad, [True] * 6)
    feed(engine, vad, [False] * frames(0.2))
    assert engine.state == TurnState.PENDING_EOT

    feed(engine, vad, [True, True, True])
    assert engine.state == TurnState.USER_SPEAKING


def test_missing_gate_uses_bounded_silence_fallback():
    engine, vad = make_engine(
        None,
        semantic_enabled=False,
        min_silence_secs=0.2,
        fallback_silence_secs=0.4,
    )
    feed(engine, vad, [True] * 6)
    events = feed(engine, vad, [False] * frames(0.4))
    committed = next(event for event in events if event.type == TURN_COMMITTED)
    assert committed.reason == REASON_TIMEOUT


def test_force_commit_and_idle_noop():
    engine, vad = make_engine(FakeGate(0.1))
    assert engine.force_commit() == []
    feed(engine, vad, [True] * 6)
    committed = engine.force_commit()
    assert committed[0].reason == REASON_MANUAL
    assert engine.force_commit() == []


def test_max_turn_bounds_audio_buffer():
    engine, vad = make_engine(FakeGate(0.1), max_turn_secs=5)
    events = feed(engine, vad, [True] * frames(5.5))
    committed = next(event for event in events if event.type == TURN_COMMITTED)
    assert committed.reason == REASON_CEILING
    assert committed.audio.size <= int(16000 * 5.1)


def test_agent_responding_ignores_audio_until_manual_interrupt():
    engine, vad = make_engine(FakeGate(0.9))
    engine.set_agent_responding(True)
    assert feed(engine, vad, [True] * 20) == []
    assert engine.state == TurnState.AGENT_RESPONDING
    engine.set_agent_responding(False)
    assert engine.state == TurnState.IDLE


def test_config_reads_and_clamps_environment(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TURN_PATIENCE_CEILING_SECS", "90")
    monkeypatch.setenv("OPENCLAW_TURN_SEMANTIC_ENABLED", "false")
    config = TurnConfig.from_env()
    assert config.patience_ceiling_secs == 20
    assert config.semantic_enabled is False


@pytest.mark.skipif(
    not list((Path(__file__).parents[1] / "models" / "smart-turn").glob("*.onnx")),
    reason="Smart Turn model has not been downloaded",
)
def test_real_smart_turn_gate():
    gate = SmartTurnGate()
    assert gate.available
    probability = gate.predict(np.zeros(16000, dtype=np.float32))
    assert probability is not None
    assert 0 <= probability <= 1
