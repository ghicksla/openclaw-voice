# OpenClaw Voice

**Open-source browser-based voice interface for AI assistants.**

Talk to your AI like you talk to Alexa — but self-hosted, private, and connected to your own agent.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)

🌐 **Website:** [openclawvoice.com](https://openclawvoice.com)

## What's Different In This Branch

This branch focuses on OpenClaw reliability and voice UX hardening. It is usable
as a standalone public fork even before any upstream merge.

- **Final-only output guard** in OpenClaw mode to prevent reasoning leakage into spoken responses.
- **Durable delayed delivery** for long-running tasks (reconnect-safe delivery state + outbox replay).
- **Reconnect-safe email copy flow** so "send a copy to my email" can complete after delayed results.
- **Voice UI improvements** for background task states, continuous mode behavior, and push-to-talk/tap interactions.
- **Semantic continuous turns** with server-owned endpointing, bounded patience, and manual interruption.
- **Reconnect-safe browser sessions** with exponential backoff and mobile audio recovery.
- **Environment path overrides** for running outside a default local OpenClaw filesystem layout.

### Fork At A Glance

| Area | Upstream `main` | This branch |
|------|------------------|-------------|
| OpenClaw stream handling | Standard stream handling | Strict final-only filtering for `openclaw:*` models |
| Long-running task delivery | Best-effort within active session | Durable delivery-state + outbox replay across reconnects |
| "Send a copy to my email" | Basic flow | Reconnect-safe queued fulfillment when final answer is delayed |
| Voice interaction UX | Baseline push-to-talk/continuous controls | Refined hold/tap behavior + clearer background/thinking statuses |
| OpenClaw filesystem assumptions | Default paths | Environment-overridable OpenClaw state/workspace paths |

If you are evaluating this fork, prioritize the OpenClaw-mode notes in:
- `docs/voice-output-sanitization.md`
- `README.md` sections on reconnect semantics and environment variables

## Features

| Feature | Description |
|---------|-------------|
| 🎤 **Local STT** | Whisper runs locally via faster-whisper. Your voice never leaves your machine. |
| 🔊 **Streaming TTS** | ElevenLabs with sentence-by-sentence streaming. Hear responses while they generate. |
| 🎯 **Voice Activity Detection** | WebRTC VAD gates speech; optional Smart Turn semantics decide when a thought is complete. |
| 🧹 **Smart Text Cleaning** | Strips markdown, hashtags, URLs before TTS. No more "hash hash". |
| 🔌 **Any AI Backend** | OpenAI, Claude, or full OpenClaw agent with memory and tools. |
| 🛡️ **Tagged-Final Output Guard** | In OpenClaw mode, streams only `<final>...</final>` content to prevent reasoning leaks. |
| 📬 **Durable Follow-up Delivery** | Persists pending background results and "send a copy to my email" requests across reconnects. |
| 🌐 **Browser-Based** | No app install. Works on desktop and mobile. |
| 🚗 **Continuous Mode** | Hands-free conversation. Auto-listens after each response. |

## Quick Start

```bash
# Clone
git clone https://github.com/Purple-Horizons/openclaw-voice.git
cd openclaw-voice

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install VAD support (better noise handling)
pip install torch torchaudio

# Configure (create .env from example)
cp .env.example .env
# Edit .env with your API keys

# Run
PYTHONPATH=. python -m src.server.main

# Open http://localhost:8765
```

## For OpenClaw Agents (Auto-Install)

One-liner for AI agents to self-install:

```bash
cd /tmp && \
git clone https://github.com/Purple-Horizons/openclaw-voice.git && \
cd openclaw-voice && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -r requirements.txt torch torchaudio && \
PYTHONPATH=. ELEVENLABS_API_KEY="$ELEVENLABS_API_KEY" OPENAI_API_KEY="$OPENAI_API_KEY" \
  nohup python -m src.server.main > /tmp/voice-server.log 2>&1 &
```

## Configuration

### Environment Variables

#### Core

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ELEVENLABS_API_KEY` | Yes* | — | ElevenLabs TTS API key |
| `OPENAI_API_KEY` | Yes* | — | OpenAI key (when not using gateway) |
| `OPENCLAW_GATEWAY_URL` | No | — | OpenClaw gateway URL |
| `OPENCLAW_GATEWAY_TOKEN` | No | — | Gateway auth token |
| `OPENCLAW_GATEWAY_AGENT_ID` | No | `main` | Gateway agent id |
| `OPENCLAW_PORT` | No | `8765` | Server port |
| `OPENCLAW_STT_MODEL` | No | `base` | Whisper model size |
| `OPENCLAW_STT_DEVICE` | No | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `OPENCLAW_REQUIRE_AUTH` | No | `false` | Require API keys for clients |
| `OPENCLAW_SESSION_GRACE_SECONDS` | No | `600` | Browser reconnect/resume window, clamped to one hour |

*One of `OPENAI_API_KEY` or `OPENCLAW_GATEWAY_URL` required.

### Semantic Continuous Turns

Continuous mode streams microphone audio until the server detects speech, waits
for a natural pause, and asks Smart Turn whether the thought is complete. If the
model is unavailable, a bounded silence timeout keeps continuous mode usable.
Hold-to-talk remains client-controlled and unchanged.

```bash
uv sync --extra semantic --extra stt
python scripts/download_models.py smart-turn
```

Turn behavior is configurable with:

- `OPENCLAW_TURN_MIN_SPEECH_FRAMES` (default `4`)
- `OPENCLAW_TURN_MIN_SILENCE_SECS` (default `1.8`)
- `OPENCLAW_TURN_FALLBACK_SILENCE_SECS` (default `2.4`)
- `OPENCLAW_TURN_RECHECK_INTERVAL_SECS` (default `2.0`)
- `OPENCLAW_TURN_PATIENCE_CEILING_SECS` (default `18`, maximum `20`)
- `OPENCLAW_TURN_MAX_TURN_SECS` (default `45`)
- `OPENCLAW_TURN_SEMANTIC_ENABLED` (default `true`)
- `OPENCLAW_TURN_SEMANTIC_THRESHOLD` (default `0.5`)

The Smart Turn inference path is adapted from
[pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) under its
BSD-2-Clause license.

### Mobile and Bluetooth Behavior

On iOS, microphone tracks are released between the user turn and TTS playback.
This lets car audio return from the low-quality hands-free call route to normal
media playback. Consequently, spoken barge-in is intentionally unavailable
while the assistant is speaking; tap the voice control (or press Space) to
interrupt immediately. The client then reacquires the microphone for the next
continuous turn.

<details>
<summary>Advanced OpenClaw path overrides (optional)</summary>

Use these only if your OpenClaw files are not in default locations.

- `OPENCLAW_WORKSPACE_ROOT` (default: auto-detected)
- `OPENCLAW_AGENTS_CONFIG` (default: `<workspace>/config/agents.json`)
- `OPENCLAW_USER_PROFILE` (default: `<workspace>/USER.md`)
- `OPENCLAW_AGENTMAIL_SEND_SCRIPT` (default: `<workspace>/skills/agentmail/scripts/send_email.py`)
- `OPENCLAW_TASK_RUNS_DB` (default: `~/.openclaw/tasks/runs.sqlite`)
- `OPENCLAW_SESSIONS_STATE` (default: `~/.openclaw/agents/main/sessions/sessions.json`)
- `OPENCLAW_VOICE_DELIVERY_DIR` (default: `~/.openclaw/voice/delivery-state`)
- `OPENCLAW_VOICE_OUTBOX_DIR` (default: `~/.openclaw/voice/outbox`)

</details>

### Whisper Model Sizes

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| `tiny` | Fastest | Fair | ~400MB | Quick testing |
| `base` | Fast | Good | ~1GB | **Default. Good balance.** |
| `small` | Medium | Better | ~2GB | Clearer transcription |
| `medium` | Slower | Great | ~5GB | Accuracy priority |
| `large-v3-turbo` | Slow | Best | ~6GB | Maximum accuracy |

### TTS Options

| Backend | Type | Quality | Latency | Notes |
|---------|------|---------|---------|-------|
| **ElevenLabs** | Cloud | Excellent | ~500ms | Default. Streaming supported. |
| Chatterbox | Local | Very Good | ~1s | MIT license, voice cloning |
| XTTS-v2 | Local | Excellent | ~1s | Voice cloning supported |
| Mock | Local | None | 0ms | For testing (silence) |

ElevenLabs uses `eleven_turbo_v2_5` for fastest response.

## OpenClaw Gateway Integration

Connect to your full OpenClaw agent (same memory, tools, and persona as text chat):

```bash
# .env
OPENCLAW_GATEWAY_URL=http://localhost:18789
OPENCLAW_GATEWAY_TOKEN=your-token
ELEVENLABS_API_KEY=your-key
```

Add to your `openclaw.json`:

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  },
  "agents": {
    "list": [
      {
        "id": "voice",
        "workspace": "/path/to/workspace",
        "model": "anthropic/claude-sonnet-4-5"
      }
    ]
  }
}
```

### Background Results and Reconnect Semantics (OpenClaw mode)

When OpenClaw tasks run longer than a foreground voice turn, the server sends an
immediate acknowledgement and tracks completion asynchronously. To keep delivery
reliable across websocket disconnects/reconnects, two local stores are used:

- `OPENCLAW_VOICE_DELIVERY_DIR`: per-session JSON state for offsets, pending email-copy requests, and last spoken answer.
- `OPENCLAW_VOICE_OUTBOX_DIR`: queued completed results that could not be spoken yet and must replay later.

On reconnect, queued results are replayed first, then normal announce polling resumes.
This makes delayed task responses and follow-up commands like "send a copy to my email"
robust even when the browser reconnects.

### Tagged-Final Sanitization (OpenClaw mode)

For OpenClaw orchestrator backends (`openclaw:*` models), streamed output is filtered
in strict mode so only text inside `<final>...</final>` is emitted to UI/TTS. This
prevents internal reasoning/tool-planning text from leaking into spoken responses.
See [`docs/voice-output-sanitization.md`](docs/voice-output-sanitization.md) for details.

## Architecture

```
┌─────────────┐   WebSocket   ┌─────────────────────────────────────┐
│   Browser   │◄────────────►│          Voice Server               │
│  (mic/spk)  │               │                                     │
└─────────────┘               │  ┌─────────┐  ┌─────┐  ┌─────────┐ │
                              │  │ Whisper │→│ AI  │→│ElevenLabs│ │
                              │  │  (STT)  │  │     │  │  (TTS)  │ │
                              │  └─────────┘  └─────┘  └─────────┘ │
                              │       ↑                     │      │
                              │    [VAD]              [streaming]  │
                              └─────────────────────────────────────┘
```

**Streaming Flow:**
1. User speaks → Whisper transcribes locally
2. AI responds (streamed) → buffer sentences
3. First sentence complete → TTS starts immediately
4. Audio streams to browser while AI continues
5. Result: ~50% faster perceived response

## HTTPS for Mobile

Mobile browsers require HTTPS for microphone access. Options:

**Tailscale Funnel (easiest):**
```bash
tailscale funnel 8765
# Access via https://your-machine.tailnet-name.ts.net
```

**nginx + Let's Encrypt:**
```nginx
server {
    listen 443 ssl;
    server_name voice.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## API

### WebSocket Protocol

Connect to `ws://localhost:8765/ws`:

```javascript
// Start hold-to-talk recording
{ "type": "start_listening", "mode": "push_to_talk", "sample_rate": 16000 }

// Start server-owned continuous recording
{ "type": "start_listening", "mode": "continuous", "sample_rate": 16000, "config": {} }

// Send audio (base64 PCM float32, 16kHz)
{ "type": "audio", "data": "base64..." }

// Stop recording
{ "type": "stop_listening" }

// Immediately cancel the current AI/TTS response
{ "type": "interrupt" }

// Receive events:
{ "type": "transcript", "text": "...", "final": true }
{ "type": "response_chunk", "text": "..." }        // Streaming text
{ "type": "audio_aac", "data": "...", "mime": "audio/aac" }    // Streaming audio
{ "type": "response_complete", "text": "..." }     // Full response
{ "type": "turn_started", "turn_id": 1 }           // Server detected speech
{ "type": "eot_pending", "turn_id": 1 }            // Semantic completion check
{ "type": "turn_committed", "turn_id": 1, "reason": "semantic" }
{ "type": "state", "state": "listening" }           // listening/thinking/speaking/idle
{ "type": "tts_cancelled" }                         // Client must flush queued playback
{ "type": "background_task_started" }              // Deferred task accepted
{ "type": "background_task_finished" }             // Deferred task result delivered
{ "type": "assistant_turn_start" }                 // Resume normal listening flow
{ "type": "vad_status", "speech_detected": true }  // VAD feedback
```

## Roadmap

- [x] WebSocket voice gateway
- [x] Whisper STT (local)
- [x] ElevenLabs TTS
- [x] Streaming TTS (sentence-by-sentence)
- [x] Voice Activity Detection (WebRTC VAD)
- [x] Semantic end-of-turn detection (optional Smart Turn model)
- [x] Text cleaning (markdown/hashtags/URLs)
- [x] Continuous conversation mode
- [x] OpenClaw gateway integration
- [ ] WebRTC for lower latency
- [ ] Voice cloning UI
- [ ] Docker support

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Local STT
- [ElevenLabs](https://elevenlabs.io) — Text-to-Speech
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) — Speech gating
- [pipecat-ai Smart Turn](https://github.com/pipecat-ai/smart-turn) — Semantic endpointing
- Built for [OpenClaw](https://openclaw.ai)

---

**Made with 🦞 by [Purple Horizons](https://purplehorizons.io)**
