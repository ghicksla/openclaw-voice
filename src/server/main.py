"""
OpenClaw Voice Server

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication
- Text-to-Speech via ElevenLabs
- Audio streaming back to browser
"""

import asyncio
import base64
from datetime import datetime
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import (
    StreamSanitizer,
    clean_for_display,
    clean_for_speech,
    extract_last_final,
    looks_like_reasoning_leak,
)


ANNOUNCE_POLL_INTERVAL_S = 5
WORKSPACE_ROOT = Path(os.getenv("OPENCLAW_WORKSPACE_ROOT") or Path(__file__).resolve().parents[4])
AGENTS_CONFIG_PATH = Path(os.getenv("OPENCLAW_AGENTS_CONFIG") or str(WORKSPACE_ROOT / "config" / "agents.json"))
USER_PROFILE_PATH = Path(os.getenv("OPENCLAW_USER_PROFILE") or str(WORKSPACE_ROOT / "USER.md"))
AGENTMAIL_SEND_SCRIPT = Path(
    os.getenv("OPENCLAW_AGENTMAIL_SEND_SCRIPT")
    or str(WORKSPACE_ROOT / "skills" / "agentmail" / "scripts" / "send_email.py")
)
TASK_RUNS_DB_PATH = Path(
    os.getenv("OPENCLAW_TASK_RUNS_DB")
    or str(Path.home() / ".openclaw" / "tasks" / "runs.sqlite")
)
SESSIONS_STATE_PATH = Path(
    os.getenv("OPENCLAW_SESSIONS_STATE")
    or str(Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json")
)
VOICE_DELIVERY_STATE_DIR = Path(
    os.getenv("OPENCLAW_VOICE_DELIVERY_DIR")
    or str(Path.home() / ".openclaw" / "voice" / "delivery-state")
)
VOICE_RESULT_OUTBOX_DIR = Path(
    os.getenv("OPENCLAW_VOICE_OUTBOX_DIR")
    or str(Path.home() / ".openclaw" / "voice" / "outbox")
)
TASK_RESULT_START = "<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>"
TASK_RESULT_END = "<<<END_UNTRUSTED_CHILD_RESULT>>>"
WORK_COMMUTE_DESTINATION = "Work destination"
WORK_COMMUTE_DEFAULT_ORIGIN = "Current location"
BACKGROUND_RESEARCH_TIMEOUT_S = 180
FOREGROUND_STANDBY_AFTER_S = 15
VOICE_RESULT_NORMAL_MAX_CHARS = 1500
VOICE_RESULT_NORMAL_MAX_SENTENCES = 10
VOICE_RESULT_DEEP_MAX_CHARS = 3000
VOICE_RESULT_DEEP_MAX_SENTENCES = 20


async def get_tasks() -> list[dict]:
    """Fetch recent OpenClaw background tasks directly from SQLite."""

    def _read_tasks() -> list[dict]:
        if not TASK_RUNS_DB_PATH.exists():
            return []

        conn = sqlite3.connect(TASK_RUNS_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                  task_id,
                  runtime,
                  source_id,
                  owner_key,
                  requester_session_key,
                  child_session_key,
                  run_id,
                  label,
                  task,
                  status,
                  delivery_status,
                  notify_policy,
                  created_at,
                  started_at,
                  ended_at,
                  last_event_at,
                  cleanup_after,
                  error,
                  progress_summary,
                  terminal_summary,
                  terminal_outcome
                FROM task_runs
                ORDER BY created_at DESC
                LIMIT 256
                """
            ).fetchall()
            return [
                {
                    "taskId": row["task_id"],
                    "runtime": row["runtime"],
                    "sourceId": row["source_id"],
                    "requesterSessionKey": row["requester_session_key"] or row["owner_key"],
                    "ownerKey": row["owner_key"],
                    "childSessionKey": row["child_session_key"],
                    "runId": row["run_id"],
                    "label": row["label"],
                    "task": row["task"],
                    "status": row["status"],
                    "delivery": row["delivery_status"],
                    "notifyPolicy": row["notify_policy"],
                    "createdAt": row["created_at"],
                    "startedAt": row["started_at"],
                    "endedAt": row["ended_at"],
                    "lastEventAt": row["last_event_at"],
                    "cleanupAfter": row["cleanup_after"],
                    "error": row["error"],
                    "progressSummary": row["progress_summary"],
                    "summary": row["terminal_summary"],
                    "terminalOutcome": row["terminal_outcome"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    try:
        return await asyncio.to_thread(_read_tasks)
    except Exception as e:
        logger.debug(f"Task poll error: {e}")
        return []


async def get_session_file_path(session_owner_key: str) -> Optional[Path]:
    """Resolve an OpenClaw session key to its JSONL transcript file."""

    def _read_session_file() -> Optional[Path]:
        if not SESSIONS_STATE_PATH.exists():
            return None
        data = json.loads(SESSIONS_STATE_PATH.read_text())
        session = data.get(session_owner_key)
        if not isinstance(session, dict):
            return None
        session_file = str(session.get("sessionFile") or "").strip()
        if not session_file:
            return None
        path = Path(session_file)
        if not path.is_absolute():
            path = SESSIONS_STATE_PATH.parent / path.name
        return path if path.exists() else None

    try:
        return await asyncio.to_thread(_read_session_file)
    except Exception as e:
        logger.debug(f"Session file lookup error for {session_owner_key}: {e}")
        return None


def _voice_state_key(session_owner_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_owner_key).strip("._-") or "default"


def _voice_delivery_state_path(session_owner_key: str) -> Path:
    return VOICE_DELIVERY_STATE_DIR / f"{_voice_state_key(session_owner_key)}.json"


def _voice_result_outbox_path(session_owner_key: str) -> Path:
    return VOICE_RESULT_OUTBOX_DIR / f"{_voice_state_key(session_owner_key)}.json"


def _normalize_pending_email_copy(raw: object) -> Optional[dict]:
    """Normalize pending email-copy request state from disk."""
    if not isinstance(raw, dict):
        return None
    to_address = str(raw.get("toAddress") or "").strip()
    if not to_address:
        return None
    min_session_offset_raw = raw.get("minSessionOffset")
    min_session_offset = (
        int(min_session_offset_raw)
        if isinstance(min_session_offset_raw, int)
        else None
    )
    requested_at_raw = raw.get("requestedAt")
    requested_at = (
        float(requested_at_raw)
        if isinstance(requested_at_raw, (int, float))
        else time.time()
    )
    require_delayed = bool(raw.get("requireDelayed", True))
    return {
        "toAddress": to_address,
        "minSessionOffset": min_session_offset,
        "requestedAt": requested_at,
        "requireDelayed": require_delayed,
    }


async def read_voice_delivery_state(session_owner_key: str) -> dict:
    """Load durable delivery state for one voice session."""

    def _read() -> dict:
        path = _voice_delivery_state_path(session_owner_key)
        if not path.exists():
            return {
                "taskIds": [],
                "messageIds": [],
                "sessionOffset": None,
                "pendingEmailCopy": None,
                "lastSpokenAnswer": None,
            }
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {
                "taskIds": [],
                "messageIds": [],
                "sessionOffset": None,
                "pendingEmailCopy": None,
                "lastSpokenAnswer": None,
            }
        return {
            "taskIds": [str(v) for v in (data.get("taskIds") or []) if str(v).strip()],
            "messageIds": [str(v) for v in (data.get("messageIds") or []) if str(v).strip()],
            "sessionOffset": data.get("sessionOffset"),
            "pendingEmailCopy": _normalize_pending_email_copy(data.get("pendingEmailCopy")),
            "lastSpokenAnswer": str(data["lastSpokenAnswer"]) if data.get("lastSpokenAnswer") else None,
        }

    try:
        return await asyncio.to_thread(_read)
    except Exception as e:
        logger.debug(f"Voice delivery-state read error for {session_owner_key}: {e}")
        return {
            "taskIds": [],
            "messageIds": [],
            "sessionOffset": None,
            "pendingEmailCopy": None,
        }


async def write_voice_delivery_state(
    session_owner_key: str,
    task_ids: set[str],
    message_ids: set[str],
    session_offset: Optional[int],
    pending_email_copy: Optional[dict],
    last_spoken_answer: Optional[str] = None,
) -> None:
    """Persist delivered IDs and last-read session offset across reconnects."""

    def _write() -> None:
        path = _voice_delivery_state_path(session_owner_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "taskIds": sorted(task_ids),
            "messageIds": sorted(message_ids),
            "sessionOffset": session_offset,
            "pendingEmailCopy": _normalize_pending_email_copy(pending_email_copy),
            "lastSpokenAnswer": last_spoken_answer or None,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.debug(f"Voice delivery-state write error for {session_owner_key}: {e}")


async def load_voice_result_outbox(session_owner_key: str) -> list[dict]:
    """Load queued background voice results waiting for replay."""

    def _read() -> list[dict]:
        path = _voice_result_outbox_path(session_owner_key)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        items: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if item_id and text:
                items.append({
                    "id": item_id,
                    "text": text,
                    "maxChars": item.get("maxChars"),
                    "maxSentences": item.get("maxSentences"),
                })
        return items

    try:
        return await asyncio.to_thread(_read)
    except Exception as e:
        logger.debug(f"Voice outbox read error for {session_owner_key}: {e}")
        return []


async def enqueue_voice_result_outbox(
    session_owner_key: str,
    text: str,
    *,
    max_chars: int = VOICE_RESULT_NORMAL_MAX_CHARS,
    max_sentences: int = VOICE_RESULT_NORMAL_MAX_SENTENCES,
) -> str:
    """Queue a durable background result for replay after reconnect."""
    cleaned = clean_for_display(text).strip() or text.strip()
    if not cleaned:
        return ""

    outbox_id = f"outbox:{os.urandom(8).hex()}"

    def _write() -> None:
        path = _voice_result_outbox_path(session_owner_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        items: list[dict] = []
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    items = [item for item in data if isinstance(item, dict)]
            except Exception:
                items = []
        items.append({
            "id": outbox_id,
            "text": cleaned,
            "maxChars": max_chars,
            "maxSentences": max_sentences,
        })
        path.write_text(json.dumps(items, indent=2))

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.debug(f"Voice outbox write error for {session_owner_key}: {e}")
        return ""
    return outbox_id


async def remove_voice_result_outbox_entries(
    session_owner_key: str,
    delivered_ids: set[str],
) -> None:
    """Remove outbox entries after they were actually delivered."""
    if not delivered_ids:
        return

    def _write() -> None:
        path = _voice_result_outbox_path(session_owner_key)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        if not isinstance(data, list):
            return
        kept = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if item_id and item_id not in delivered_ids:
                kept.append(item)
        if kept:
            path.write_text(json.dumps(kept, indent=2))
        else:
            path.unlink(missing_ok=True)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.debug(f"Voice outbox cleanup error for {session_owner_key}: {e}")


async def read_new_session_events(
    session_file_path: Path,
    offset: int,
) -> tuple[int, list[dict]]:
    """Read replayable delivery events from a session JSONL file."""

    def _read_events() -> tuple[int, list[dict]]:
        if not session_file_path.exists():
            return offset, []

        with session_file_path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read()
            next_offset = fh.tell()

        if not data:
            return next_offset, []

        events: list[dict] = []
        for raw_line in data.decode("utf-8", errors="replace").splitlines():
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if item.get("type") != "message":
                continue
            message = item.get("message") or {}
            role = message.get("role")
            content_items = message.get("content") or []
            text = "\n".join(
                str(part.get("text") or "")
                for part in content_items
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            if not text:
                continue

            if role == "user":
                provenance = message.get("provenance") or {}
                source_tool = str(provenance.get("sourceTool") or "")
                if (
                    source_tool == "subagent_announce"
                    or TASK_RESULT_START in text
                    or "sourceTool=subagent_announce" in text
                ):
                    events.append({
                        "id": item.get("id"),
                        "kind": "announce",
                        "text": text,
                    })
                continue

            if role == "assistant":
                final_text = extract_last_final(text)
                if final_text:
                    events.append({
                        "id": item.get("id"),
                        "kind": "assistant_final",
                        "text": final_text,
                    })

        return next_offset, events

    try:
        return await asyncio.to_thread(_read_events)
    except Exception as e:
        logger.debug(f"Session event read error for {session_file_path}: {e}")
        return offset, []


async def read_assistant_final_events_after(
    session_file_path: Optional[Path],
    offset: int,
) -> list[dict]:
    """Return assistant ``<final>`` events appended after ``offset``."""
    if session_file_path is None:
        return []
    _, events = await read_new_session_events(session_file_path, offset)
    return [event for event in events if event.get("kind") == "assistant_final"]


async def snapshot_session_offset(session_owner_key: str) -> tuple[Optional[Path], int]:
    """Resolve the session JSONL path and capture its current end-of-file offset.

    Used by ``process_transcript`` to bracket a single orchestrator turn so we
    can extract the ``<final>`` block written during this turn rather than
    relying on the gateway's tag-stripped flat-prose response body.
    """
    path = await get_session_file_path(session_owner_key)
    if not path:
        return None, 0
    try:
        offset = await asyncio.to_thread(lambda: path.stat().st_size if path.exists() else 0)
    except Exception as e:
        logger.debug(f"Session offset snapshot error for {path}: {e}")
        return path, 0
    return path, offset


async def read_assistant_finals_after(
    session_file_path: Optional[Path],
    offset: int,
) -> list[str]:
    """Return ``<final>`` text from assistant messages appended after ``offset``.

    The orchestrator can emit multiple assistant messages in a single turn
    (one per agent step). Only the final agent step contains the user-visible
    ``<final>`` block, so the caller usually wants the last entry.
    """
    events = await read_assistant_final_events_after(session_file_path, offset)
    return [str(event.get("text") or "").strip() for event in events if str(event.get("text") or "").strip()]


async def read_recent_assistant_final_texts(
    session_file_path: Optional[Path],
    *,
    before_offset: Optional[int] = None,
    limit: int = 6,
) -> list[str]:
    """Read recent user-visible assistant finals from the current voice session."""

    def _read_finals() -> list[str]:
        if session_file_path is None or not session_file_path.exists():
            return []

        with session_file_path.open("rb") as fh:
            if before_offset is not None:
                data = fh.read(max(0, before_offset))
            else:
                data = fh.read()

        finals: list[str] = []
        for raw_line in data.decode("utf-8", errors="replace").splitlines():
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if item.get("type") != "message":
                continue
            message = item.get("message") or {}
            if message.get("role") != "assistant":
                continue
            text = "\n".join(
                str(part.get("text") or "")
                for part in message.get("content") or []
                if isinstance(part, dict) and part.get("type") == "text"
            )
            extracted = extract_last_final(text)
            final_text = extracted.strip() if extracted else ""
            if final_text:
                finals.append(final_text)

        return finals[-limit:]

    try:
        return await asyncio.to_thread(_read_finals)
    except Exception as e:
        logger.debug(f"Recent assistant final read error for {session_file_path}: {e}")
        return []


def extract_task_result_text(task_payload: str) -> str:
    """Extract the finished child result from an announce payload."""
    result_text = task_payload
    if TASK_RESULT_START in task_payload and TASK_RESULT_END in task_payload:
        result_text = task_payload.split(TASK_RESULT_START, 1)[1].split(TASK_RESULT_END, 1)[0].strip()

    if result_text.startswith("{") or result_text.startswith("["):
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict):
                result_text = (
                    parsed.get("message")
                    or parsed.get("result")
                    or parsed.get("text")
                    or parsed.get("summary")
                    or result_text
                )
            elif isinstance(parsed, str):
                result_text = parsed
        except json.JSONDecodeError:
            pass

    return result_text.strip()


def split_text_for_tts(text: str, max_chars: int = 320) -> list[str]:
    """Split text into speech-friendly chunks."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return []

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not parts:
        return [cleaned[:max_chars]]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def cap_voice_result(
    text: str,
    max_chars: int = VOICE_RESULT_NORMAL_MAX_CHARS,
    max_sentences: int = VOICE_RESULT_NORMAL_MAX_SENTENCES,
) -> str:
    """Keep delayed voice follow-ups short enough to be useful aloud."""
    cleaned = clean_for_display(text).strip()
    if not cleaned:
        return ""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if sentences:
        capped = " ".join(sentences[:max_sentences]).strip()
        if len(capped) <= max_chars:
            return capped

    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}."


def has_deep_analysis_intent(text: str) -> bool:
    """Return true when the user explicitly asks for a deeper-than-normal answer."""
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "deep thinking",
            "deep analysis",
            "deeper analysis",
            "deep dive",
            "go deep",
            "go deeper",
            "think deeply",
            "analyze deeply",
            "comprehensive analysis",
            "thorough analysis",
            "detailed analysis",
            "exhaustive analysis",
        )
    )


def voice_result_budget_for_prompt(text: str) -> tuple[int, int]:
    """Choose delayed voice-result size from the user's requested depth."""
    if has_deep_analysis_intent(text):
        return VOICE_RESULT_DEEP_MAX_CHARS, VOICE_RESULT_DEEP_MAX_SENTENCES
    return VOICE_RESULT_NORMAL_MAX_CHARS, VOICE_RESULT_NORMAL_MAX_SENTENCES


def _runtime_env_value(name: str) -> str:
    """Read a runtime env var from process env or OpenClaw config env."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        cfg = json.loads(config_path.read_text())
        return str((cfg.get("env") or {}).get(name) or "").strip()
    except Exception:
        return ""


def is_work_commute_request(text: str) -> bool:
    """Return true for voice requests about the user's normal drive to work."""
    lower = text.lower()
    workish = any(term in lower for term in ("to work", "commute", "route to work", "traffic to work"))
    routeish = any(term in lower for term in ("route", "traffic", "accident", "avoid", "leave", "drive"))
    return workish and routeish


def is_local_time_request(text: str) -> bool:
    """Return true for direct requests asking for the current local day/date/time."""
    lower = text.lower().strip()
    direct_patterns = (
        r"\bwhat(?:'s| is)?\s+the\s+time\b",
        r"\bwhat\s+time\s+is\s+it\b",
        r"\bwhat(?:'s| is)?\s+the\s+(day|date)\b",
        r"\bwhat\s+(day|date)\s+is\s+it\b",
        r"\bwhat(?:'s| is)?\s+the\s+(day|date)\s+and\s+time\b",
        r"\bwhat(?:'s| is)?\s+the\s+time\s+and\s+(day|date)\b",
        r"\bwhat\s+(day|date)\s+and\s+time\s+is\s+it\b",
        r"\bwhat\s+time\s+and\s+(day|date)\s+is\s+it\b",
        r"\bcurrent\s+time\b",
        r"\bcurrent\s+(day|date)\b",
        r"\btoday'?s\s+(day|date)\b",
        r"\btell\s+me\s+the\s+time\b",
        r"\btell\s+me\s+the\s+(day|date)\b",
        r"\btell\s+me\s+the\s+(day|date)\s+and\s+time\b",
        r"\btime\s+is\s+it\b",
    )
    return any(re.search(pattern, lower) for pattern in direct_patterns)


def build_local_time_response() -> str:
    """Return a deterministic local-time voice response."""
    now_local = datetime.now().astimezone()
    spoken_time = now_local.strftime("%I:%M %p").lstrip("0")
    spoken_month_year = now_local.strftime("%A, %B")
    tz_name = now_local.tzname() or "local time"
    return f"It's {spoken_month_year} {now_local.day}, {now_local.year} at {spoken_time} {tz_name}."


def detect_voice_intents(text: str) -> set[str]:
    """Classify primary voice intents with regex fast-path + semantic fallback."""
    intents: set[str] = set()
    lower = text.lower().strip()
    tokens = set(re.findall(r"[a-z']+", lower))

    if is_send_copy_to_email_request(text):
        intents.add("email_copy")
    elif lower:
        # Semantic fallback so we do not need to enumerate every phrasing.
        email_action_terms = {
            "send", "email", "mail", "forward", "share", "deliver",
        }
        email_target_terms = {
            "email", "gmail", "inbox", "mailbox",
        }
        email_object_terms = {
            "it", "this", "that", "copy", "summary", "answer", "response", "recap", "result",
        }
        if (
            tokens.intersection(email_action_terms)
            and tokens.intersection(email_target_terms)
            and (
                tokens.intersection(email_object_terms)
                or "to my" in lower
                or "send me" in lower
                or "for me" in lower
            )
        ):
            intents.add("email_copy")

    if is_work_commute_request(text):
        intents.add("commute")

    if is_background_research_request(text):
        intents.add("background_research")
    elif lower:
        # Fallback intent for natural variants like:
        # "can you research X and let me know when done".
        async_markers = (
            "background",
            "get back to me",
            "when finished",
            "when done",
            "later",
            "follow up",
            "let me know",
        )
        research_tokens = {
            "research", "analyze", "analysis", "investigate", "look", "check", "deep", "dive",
        }
        if tokens.intersection(research_tokens) and any(marker in lower for marker in async_markers):
            intents.add("background_research")

    return intents


def should_delay_email_copy_request(text: str, *, has_background_work: bool = False) -> bool:
    """True when an email-copy request should wait for a delayed/background answer."""
    lower = text.lower()
    delayed_markers = (
        "when done",
        "when it's done",
        "when its done",
        "when finished",
        "when ready",
        "when complete",
        "when completed",
        "once done",
        "once finished",
        "as soon as",
        "after it finishes",
        "after it is finished",
        "after it's finished",
    )
    return has_background_work or any(marker in lower for marker in delayed_markers)


def extract_compound_email_task_text(text: str) -> Optional[str]:
    """Return the task part from "do X and email it when done" style requests."""
    match = re.search(
        r"\b(?:and|then|also)?\s*(?:send|email|mail|forward)\b.{0,80}\b(?:email|gmail|inbox|mailbox)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match or match.start() < 8:
        return None

    task_text = re.sub(r"\s+(?:and|then|also)\s*$", "", text[: match.start()].strip(), flags=re.IGNORECASE)
    if len(task_text.split()) < 3:
        return None
    return task_text


def is_send_copy_to_email_request(text: str) -> bool:
    """Return true for explicit voice follow-ups asking to email the prior answer."""
    lower = text.lower()
    explicit_patterns = (
        r"\b(send|email|mail|forward)\b.{0,40}\b(copy|summary)\b.{0,40}\b(email|gmail|my inbox)\b",
        r"\b(send|email|mail|forward)\b.{0,24}\bthat\b.{0,40}\b(email|gmail|my inbox)\b",
        r"\b(send|email|mail|forward)\b.{0,24}\b(it|this)\b.{0,40}\b(to\b.{0,24})?(email|gmail|my inbox)\b",
        r"\bsend a copy of that\b",
        r"\bemail that to my gmail\b",
        r"\bemail it to my gmail\b",
        r"\bemail this to my gmail\b",
    )
    if any(re.search(pattern, lower) for pattern in explicit_patterns):
        return True
    return False


def load_agentmail_inbox_address() -> str:
    try:
        config = json.loads(AGENTS_CONFIG_PATH.read_text())
        inbox = str(config.get("inboxAddress") or "").strip()
        if inbox:
            return inbox
    except Exception as e:
        logger.debug(f"Could not read AgentMail inbox address: {e}")
    return "assistant@agentmail.to"


def load_primary_email_address() -> str:
    try:
        text = USER_PROFILE_PATH.read_text()
    except Exception as e:
        logger.debug(f"Could not read primary user email: {e}")
        return "user@example.com"

    match = re.search(r"Email \(primary\):\*\*\s*([^\s]+)", text)
    if match:
        return match.group(1).strip()
    return "user@example.com"


def build_email_copy_body(final_texts: list[str]) -> str:
    """Build a plain-text email from recent spoken answers."""
    usable: list[str] = []
    for text in final_texts:
        cleaned = clean_for_display(text).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if "sent" in lowered and ("email" in lowered or "inbox" in lowered):
            continue
        if (
            "working on that" in lowered
            or "please stand by" in lowered
            or "working on it" in lowered
            or "respond when finished" in lowered
        ):
            continue
        if cleaned not in usable:
            usable.append(cleaned)

    if not usable:
        return ""

    body_lines = ["Hello,", "", "Here is the summary you requested:", ""]
    items = usable[-6:]
    if len(items) == 1:
        body_lines.append(items[0])
    else:
        for index, item in enumerate(items, start=1):
            body_lines.append(f"{index}. {item}")
    body_lines.extend(["", "Best,", "Assistant"])
    return "\n".join(body_lines)


def _pick_recent_copy_candidate(final_texts: list[str]) -> str:
    """Pick the newest candidate answer that is safe to email."""
    for candidate in reversed(final_texts):
        cleaned = clean_for_display(candidate).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in {
            "i do not have a finished answer to copy yet. i will send it as soon as it is ready.",
            "i could not send that email cleanly. please try again in a minute.",
            "done. i sent a clean copy to your email inbox.",
            "i sent that summary to your email inbox.",
            "i also sent that summary to your email inbox.",
        }:
            continue
        if (
            "please stand by" in lowered
            or "working on that" in lowered
            or "working on it" in lowered
            or "respond when finished" in lowered
        ):
            continue
        if is_nonfinal_background_result(cleaned):
            continue
        return cleaned
    return ""


def _make_pending_email_copy_request(
    *,
    to_address: str,
    min_session_offset: Optional[int],
    require_delayed: bool = True,
) -> dict:
    return {
        "toAddress": to_address,
        "minSessionOffset": min_session_offset,
        "requestedAt": time.time(),
        "requireDelayed": require_delayed,
    }


def _can_fulfill_pending_email_copy(
    pending_request: Optional[dict],
    *,
    source: str,
    source_offset: Optional[int],
    text: str,
) -> bool:
    pending = _normalize_pending_email_copy(pending_request)
    if not pending:
        return False
    cleaned = clean_for_display(text).strip()
    if not cleaned:
        return False
    if is_nonfinal_background_result(cleaned):
        return False
    if pending.get("requireDelayed") and source != "delayed":
        return False
    min_offset = pending.get("minSessionOffset")
    if (
        isinstance(min_offset, int)
        and isinstance(source_offset, int)
        and source_offset < min_offset
    ):
        return False
    return True


async def send_agentmail_plain_text(
    *,
    to_address: str,
    subject: str,
    text: str,
) -> bool:
    """Send AgentMail without shell interpolation, preserving literal dollars."""
    if not AGENTMAIL_SEND_SCRIPT.exists():
        logger.error(f"AgentMail send script not found: {AGENTMAIL_SEND_SCRIPT}")
        return False

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(AGENTMAIL_SEND_SCRIPT),
        "--inbox",
        load_agentmail_inbox_address(),
        "--to",
        to_address,
        "--subject",
        subject,
        "--text",
        text,
        cwd=str(WORKSPACE_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        logger.error("AgentMail send timed out")
        return False

    if proc.returncode != 0:
        logger.error(
            "AgentMail send failed with code {}: {}{}",
            proc.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
        return False
    return True


def is_background_research_request(text: str) -> bool:
    """Heuristic for voice prompts that need durable async research."""
    lower = text.lower()
    explicit_async = any(
        phrase in lower
        for phrase in (
            "background research",
            "spawn multiple agents",
            "multiple agents",
            "sub agents",
            "subagents",
            "get back to me when they're done",
            "get back to me when theyre done",
            "get back to me later",
            "deep research",
            "research this deeply",
            "look into",
            "check into",
        )
    )
    research_terms = sum(
        1
        for phrase in (
            "research",
            "analysis",
            "analyst",
            "outlook",
            "pros",
            "cons",
            "risk",
            "risks",
            "sources",
            "objective",
            "unbiased",
        )
        if phrase in lower
    )
    return explicit_async or (len(lower) >= 180 and research_terms >= 2)


def is_placeholder_gateway_response(text: str) -> bool:
    """True for shim/protocol placeholders that are not a finished answer."""
    sample = clean_for_display(text).strip().lower()
    return sample in {
        "",
        "no response",
        "no response from openclaw.",
        "no response from openclaw",
    }


def is_nonfinal_background_result(text: str) -> bool:
    """Suppress raw subagent/status payloads until the parent final arrives."""
    sample = clean_for_display(text).strip().lower()
    if is_placeholder_gateway_response(text):
        return True
    return any(
        marker in sample
        for marker in (
            "[partial progress:",
            "partial progress:",
            "before timeout",
            "timed out",
            "model idle timeout",
            "request timed out",
            "did not produce a response before",
            "### ",
            "#### ",
            "performance summary",
            "analyst outlook",
            "consensus rating:",
            "price targets:",
        )
    )


async def _run_json_command(args: list[str], env: dict[str, str], timeout: int = 35) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(WORKSPACE_ROOT),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"status": "error", "message": "The traffic lookup timed out."}

    output = stdout.decode("utf-8", errors="replace").strip()
    if not output:
        err = stderr.decode("utf-8", errors="replace").strip()
        return {"status": "error", "message": err or "The traffic lookup returned no data."}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"status": "error", "message": output[:300]}


def _summarize_tomtom_commute(data: dict) -> str:
    if data.get("status") != "success" or not data.get("routes"):
        return str(data.get("message") or "I could not get traffic data for that route.")
    route = data["routes"][0]
    minutes = round(float(route.get("travel_time_min") or 0))
    delay = round(float(route.get("traffic_delay_min") or 0))
    congestion = route.get("congestion") or "unknown"
    if delay > 0:
        return (
            f"The fastest route is about {minutes} minutes right now, "
            f"with {delay} minutes of {congestion} traffic delay. Leave now if you want a buffer."
        )
    return f"The fastest route is about {minutes} minutes right now with light traffic. Leave now if you want a buffer."


def _summarize_google_commute(data: dict) -> str:
    if data.get("error"):
        return str(data.get("error"))
    duration = data.get("duration_in_traffic") or data.get("duration") or "unknown"
    static_duration = data.get("static_duration")
    delay_phrase = ""
    if duration and static_duration and duration != static_duration:
        delay_phrase = f", compared with {static_duration} without traffic"
    return f"The Google Maps estimate is {duration}{delay_phrase}. I do not have accident details unless TomTom is configured."


async def resolve_work_commute_response(transcript: str) -> Optional[str]:
    """Fast-path commute requests so voice doesn't start a long build/tool task."""
    if not is_work_commute_request(transcript):
        return None

    origin = _runtime_env_value("OPENCLAW_COMMUTE_ORIGIN") or WORK_COMMUTE_DEFAULT_ORIGIN
    tomtom_key = _runtime_env_value("TOMTOM_API_KEY")
    google_key = _runtime_env_value("GOOGLE_API_KEY") or _runtime_env_value("GOOGLE_MAPS_API_KEY")
    env = {**os.environ}

    if tomtom_key:
        env["TOMTOM_API_KEY"] = tomtom_key
        data = await _run_json_command(
            [
                "python3",
                "skills/openclaw-commute-traffic/scripts/check_traffic.py",
                "--origin",
                origin,
                "--destination",
                WORK_COMMUTE_DESTINATION,
            ],
            env,
        )
        return _summarize_tomtom_commute(data)

    if google_key:
        env["GOOGLE_API_KEY"] = google_key
        data = await _run_json_command(
            [
                "python3",
                "skills/google-maps/lib/map_helper.py",
                "distance",
                origin,
                WORK_COMMUTE_DESTINATION,
                "--depart=now",
                "--traffic=best_guess",
            ],
            env,
        )
        return _summarize_google_commute(data)

    return (
        "I can set this up, but I need a Google Maps API key or TomTom API key first. "
        "An existing Google Workspace login does not provide Maps traffic API access."
    )


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None  # Path to voice sample for cloning
    elevenlabs_voice_id: Optional[str] = None  # ElevenLabs voice ID
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # OpenClaw Gateway (auto-detected from OPENCLAW_GATEWAY_URL + TOKEN)
    openclaw_gateway_url: Optional[str] = None
    openclaw_gateway_token: Optional[str] = None
    
    # Audio
    sample_rate: int = 16000
    
    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.1.0")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad
    
    logger.info("Initializing OpenClaw Voice server...")
    
    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")
    
    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
    )
    
    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    voice_id = settings.elevenlabs_voice_id or os.getenv("ELEVENLABS_VOICE_ID")
    tts = ChatterboxTTS(
        voice_sample=settings.tts_voice,
        voice_id=voice_id,
    )
    
    # Initialize AI backend
    # Auto-detect OpenClaw gateway
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    
    if gateway_url and gateway_token:
        # Use OpenClaw gateway (connects to Aria!)
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url}")
        agent_id = (os.getenv("OPENCLAW_GATEWAY_AGENT_ID") or "main").strip() or "main"
        backend = AIBackend(
            backend_type="openai",  # Gateway speaks OpenAI API
            url=f"{gateway_url}/v1",
            model=f"openclaw:{agent_id}",  # Choose agent via model string
            api_key=gateway_token,
            system_prompt=(
                "Voice mode: your reply is read aloud by TTS. "
                "Default to one short sentence. Use a second short sentence only when needed. "
                "If the user wants more, wait for them to ask. "
                "Use plain spoken English only — no markdown, no tables, no lists, no code blocks, no asterisks. "
                "Speak numbers and times as words."
            ),
        )
    else:
        # Fallback to direct OpenAI
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()
    
    logger.info("✅ OpenClaw Voice server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse("src/client/index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """
    Create a new API key (requires master key).
    
    curl -X POST "http://localhost:8765/api/keys?name=myapp&tier=pro" \
         -H "x-master-key: YOUR_MASTER_KEY"
    """
    # Verify master key
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            # Also check if it's a valid master-tier key
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}
    
    from .auth import PRICING_TIERS
    
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    
    tier_config = PRICING_TIERS[tier]
    
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    
    return {
        "api_key": plaintext_key,  # Only shown once!
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """
    Get usage stats for an API key.
    
    curl "http://localhost:8765/api/usage?api_key=ocv_xxx"
    """
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    
    return token_manager.get_usage(key)


def _resolve_display_model() -> str:
    """Read the active model from openclaw.json for display in the voice UI."""
    try:
        import json as _json
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        if config_path.exists():
            cfg = _json.loads(config_path.read_text())
            return cfg.get("agents", {}).get("defaults", {}).get("model", "")
    except Exception:
        pass
    return ""


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    # Check for API key in query params or headers
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    
    api_key: Optional[APIKey] = None
    
    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        # Dev mode - allow all
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")
    
    await websocket.accept()
    
    # Stable browser session key for context continuity (and OpenClaw session routing).
    session_id = websocket.query_params.get("session_id") or "default"
    agent_id = "main"
    if backend and getattr(backend, "model", "").startswith("openclaw:"):
        agent_id = backend.model.split(":", 1)[1] or "main"
    session_owner_key = f"agent:{agent_id}:openai-user:{session_id}"

    # Send session info on connect
    model_name = (backend.model or "unknown").replace("openclaw:", "agent:")
    display_model = _resolve_display_model() if model_name.startswith("agent:") else model_name
    await websocket.send_json({
        "type": "session_info",
        "model": model_name,
        "display_model": display_model,
        "sessionId": session_id,
    })

    audio_buffer = []
    is_listening = False
    session_start = None
    delivered_announce_task_ids: set[str] = set()
    delivered_session_message_ids: set[str] = set()
    background_research_tasks: set[asyncio.Task] = set()
    voice_result_max_chars = VOICE_RESULT_NORMAL_MAX_CHARS
    voice_result_max_sentences = VOICE_RESULT_NORMAL_MAX_SENTENCES
    pending_email_copy_request: Optional[dict] = None
    last_spoken_answer: Optional[str] = None

    def is_session_announce_task(task: dict) -> bool:
        task_id = task.get("taskId") or task.get("id")
        source_id = task.get("sourceId") or ""
        owner_key = task.get("ownerKey") or task.get("requesterSessionKey")
        status = task.get("status")
        return bool(
            task_id
            and source_id.startswith("announce:v1:")
            and owner_key == session_owner_key
            and status == "succeeded"
        )

    async def persist_delivery_state(
        *,
        session_offset: Optional[int],
    ) -> None:
        await write_voice_delivery_state(
            session_owner_key,
            delivered_announce_task_ids,
            delivered_session_message_ids,
            session_offset,
            pending_email_copy_request,
            last_spoken_answer,
        )

    async def maybe_send_pending_email_copy(
        finalized_text: str,
        *,
        source: str,
        source_offset: Optional[int] = None,
    ) -> Optional[str]:
        nonlocal pending_email_copy_request

        pending = _normalize_pending_email_copy(pending_email_copy_request)
        if not pending:
            return None

        if not _can_fulfill_pending_email_copy(
            pending,
            source=source,
            source_offset=source_offset,
            text=finalized_text,
        ):
            return None

        cleaned = clean_for_display(finalized_text).strip()
        body = build_email_copy_body([cleaned]) or cleaned
        ok = await send_agentmail_plain_text(
            to_address=str(pending.get("toAddress") or load_primary_email_address()),
            subject="OpenClaw summary",
            text=body,
        )
        if not ok:
            return None

        pending_email_copy_request = None
        await persist_delivery_state(session_offset=source_offset)
        return "I sent that summary to your email inbox."

    def track_background_task(task: asyncio.Task) -> None:
        background_research_tasks.add(task)
        task.add_done_callback(lambda finished: background_research_tasks.discard(finished))

    async def deliver_background_failure(message: str, *, source_offset: Optional[int] = None) -> None:
        """Tell the client a background turn failed and clear pending delivery state."""
        nonlocal pending_email_copy_request, last_spoken_answer
        display_text = clean_for_display(message).strip()
        if not display_text:
            return

        pending_email_copy_request = None
        last_spoken_answer = display_text
        await persist_delivery_state(session_offset=source_offset)

        await websocket.send_json({"type": "background_task_finished"})
        await websocket.send_json({"type": "assistant_turn_start"})
        await websocket.send_json({
            "type": "response_chunk",
            "text": display_text,
        })

        seq = 0
        for chunk in split_text_for_tts(display_text):
            aac_bytes = await tts.synthesize_aac(chunk)
            if not aac_bytes:
                continue
            audio_b64 = base64.b64encode(aac_bytes).decode()
            await websocket.send_json({
                "type": "audio_aac",
                "data": audio_b64,
                "seq": seq,
                "mime": getattr(tts, "mime_type", "audio/aac"),
            })
            seq += 1

        await asyncio.sleep(0.5)
        await websocket.send_json({
            "type": "response_complete",
            "text": display_text,
        })
        logger.info(f"Delivered background failure: {display_text[:100]}...")

    async def run_background_research(transcript: str) -> None:
        """Continue a long voice research turn off the live request path.

        The main turn speaks an immediate acknowledgment and returns. This
        detached task does the actual work with a larger timeout. If the
        orchestrator writes a normal <final> block to the session JSONL, the
        announce watcher will replay it live or after reconnect. If not, we
        queue a durable outbox item as a fallback.
        """
        max_chars, max_sentences = voice_result_budget_for_prompt(transcript)
        session_path, session_offset = await snapshot_session_offset(session_owner_key)
        try:
            response_text = await asyncio.wait_for(
                backend.chat(transcript, user_key=session_id),
                timeout=BACKGROUND_RESEARCH_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Background voice research exceeded {}; clearing pending state.",
                BACKGROUND_RESEARCH_TIMEOUT_S,
            )
            await deliver_background_failure(
                "I couldn't finish that background research in time. Please try again with a narrower request.",
                source_offset=session_offset,
            )
            return
        except Exception as e:
            logger.error(f"Background voice research failed: {e}")
            await deliver_background_failure(
                "I couldn't finish that background research cleanly. Please try again in a minute.",
                source_offset=session_offset,
            )
            return

        if session_path is None:
            session_path = await get_session_file_path(session_owner_key)
        final_events = await read_assistant_final_events_after(session_path, session_offset)
        if final_events:
            return

        if is_placeholder_gateway_response(response_text):
            logger.info(
                "Background voice research returned placeholder; clearing pending state."
            )
            await deliver_background_failure(
                "I couldn't get a finished background answer for that. Please try again with a narrower request.",
                source_offset=session_offset,
            )
            return

        cleaned = clean_for_display(response_text).strip()
        if cleaned and not looks_like_reasoning_leak(response_text):
            await enqueue_voice_result_outbox(
                session_owner_key,
                cap_voice_result(
                    cleaned,
                    max_chars=max_chars,
                    max_sentences=max_sentences,
                ),
                max_chars=max_chars,
                max_sentences=max_sentences,
            )
            return

        logger.warning(
            "Background voice research returned no clean final; clearing pending state."
        )
        await deliver_background_failure(
            "I couldn't get a clean background answer for that. Please try again with a narrower request.",
            source_offset=session_offset,
        )

    async def process_transcript(transcript: str, *, echo_transcript: bool = True):
        """Stream AI text to client; synthesize AAC per-sentence and send immediately."""
        nonlocal voice_result_max_chars, voice_result_max_sentences
        nonlocal pending_email_copy_request, last_spoken_answer
        if not transcript.strip():
            return
        voice_result_max_chars, voice_result_max_sentences = voice_result_budget_for_prompt(transcript)
        intents = detect_voice_intents(transcript)
        wants_email_copy = "email_copy" in intents
        wants_background_research = "background_research" in intents
        routed_transcript = transcript
        compound_email_task = extract_compound_email_task_text(transcript)
        if compound_email_task:
            routed_transcript = compound_email_task
            routed_intents = detect_voice_intents(routed_transcript)
            wants_background_research = "background_research" in routed_intents

        if echo_transcript:
            await websocket.send_json({
                "type": "transcript",
                "text": transcript,
                "final": True,
            })
        logger.info(f"Transcript: {transcript}")

        # TTS worker: pulls sentences off a queue, synthesizes AAC, sends audio.
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_worker():
            seq = 0
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                speech_text = clean_for_speech(sentence)
                if not speech_text:
                    continue
                logger.debug(f"Synthesizing audio [{seq}]: {speech_text[:60]}...")
                aac_bytes = await tts.synthesize_aac(speech_text)
                if aac_bytes:
                    audio_b64 = base64.b64encode(aac_bytes).decode()
                    await websocket.send_json({
                        "type": "audio_aac",
                        "data": audio_b64,
                        "seq": seq,
                        "mime": getattr(tts, "mime_type", "audio/aac"),
                    })
                seq += 1

        tts_task = asyncio.create_task(tts_worker())

        SENTENCE_ENDS = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        full_response = ""
        sentence_buffer = ""
        foreground_standby_started = False

        # The OpenClaw orchestrator publishes a strict tagged-reasoning contract
        # (system-prompt-BIKbdIsV.js): only text inside <final>...</final> is
        # user-visible. Enforce it on the consumer side because the gateway's
        # OpenAI-compat stream sometimes leaks the surrounding reasoning when
        # native thinking and tagged-final outputs are mixed (Gemini 3 Flash).
        is_orchestrator_backend = bool(
            backend and getattr(backend, "model", "").startswith("openclaw:")
        )
        sanitizer = StreamSanitizer(strict_final=is_orchestrator_backend)

        async def emit_filtered(text: str) -> None:
            nonlocal full_response, sentence_buffer
            if not text:
                return
            full_response += text
            sentence_buffer += text
            await websocket.send_json({
                "type": "response_chunk",
                "text": text,
            })
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
                    await tts_queue.put(sentence)
            # Also flush a sentence that ends with punctuation even when there is
            # no trailing whitespace yet (e.g. "I'll respond when finished.").
            stripped_tail = sentence_buffer.strip()
            if stripped_tail and stripped_tail.endswith((".", "!", "?")):
                sentence_buffer = ""
                await tts_queue.put(stripped_tail)

        if is_local_time_request(transcript):
            await emit_filtered(build_local_time_response())
            raw_stream_chars = len(full_response)
        elif wants_email_copy and compound_email_task:
            session_path, session_offset = await snapshot_session_offset(session_owner_key)
            pending_email_copy_request = _make_pending_email_copy_request(
                to_address=load_primary_email_address(),
                min_session_offset=session_offset,
                require_delayed=wants_background_research or bool(background_research_tasks),
            )
            await persist_delivery_state(session_offset=session_offset)
            wants_email_copy = False

        if wants_email_copy:
            session_path, session_offset = await snapshot_session_offset(session_owner_key)
            delay_email_copy = should_delay_email_copy_request(
                transcript,
                has_background_work=bool(background_research_tasks),
            )
            recent_finals = await read_recent_assistant_final_texts(
                session_path,
                before_offset=session_offset,
            )
            candidate = "" if delay_email_copy else _pick_recent_copy_candidate(recent_finals)
            # Fall back to the in-memory last spoken answer (survives reconnects
            # because it's also stored in the delivery state).
            if not candidate and last_spoken_answer and not delay_email_copy:
                candidate = _pick_recent_copy_candidate([last_spoken_answer])
            if candidate:
                body = build_email_copy_body([candidate]) or candidate
                sent_ok = await send_agentmail_plain_text(
                    to_address=load_primary_email_address(),
                    subject="OpenClaw summary",
                    text=body,
                )
                if sent_ok:
                    pending_email_copy_request = None
                    await persist_delivery_state(session_offset=session_offset)
                    await emit_filtered("Done. I sent a clean copy to your email inbox.")
                else:
                    await emit_filtered("I could not send that email cleanly. Please try again in a minute.")
            else:
                pending_email_copy_request = _make_pending_email_copy_request(
                    to_address=load_primary_email_address(),
                    min_session_offset=session_offset,
                    require_delayed=delay_email_copy,
                )
                await persist_delivery_state(session_offset=session_offset)
                queued_message = (
                    "I queued it and will send it when the background answer is ready."
                    if delay_email_copy
                    else "I do not have a finished answer to copy yet. I queued it and will send it as soon as that answer is ready."
                )
                await emit_filtered(queued_message)
            raw_stream_chars = len(full_response)
        elif is_orchestrator_backend and wants_background_research:
            await websocket.send_json({"type": "background_task_started"})
            await emit_filtered(
                "Working on it, I'll respond when finished."
            )
            background_task = asyncio.create_task(run_background_research(routed_transcript))
            track_background_task(background_task)
        else:
            commute_response = await resolve_work_commute_response(routed_transcript)
            if commute_response:
                await emit_filtered(commute_response)
                raw_stream_chars = len(commute_response)
            else:
                raw_stream_chars = 0

            if commute_response:
                pass
            elif is_orchestrator_backend:
            # The OpenClaw gateway has two failure modes for native-thinking
            # Gemini models: its streaming OpenAI-compat path emits an empty
            # delta then [DONE], and its non-streaming path strips the
            # <think>/<final> markers and returns the entire reasoning blob
            # as flat prose in choices[0].message.content. Either way the
            # gateway response text is unsafe to speak directly.
            #
            # The orchestrator's session JSONL still preserves the tagged
            # contract correctly, so we bracket the turn with a byte-offset
            # snapshot, run the gateway call, and pull the authoritative
            # <final>...</final> block from the new tail of the JSONL.
                session_path, session_offset = await snapshot_session_offset(
                    session_owner_key
                )
                response_task = asyncio.create_task(
                    backend.chat(routed_transcript, user_key=session_id)
                )
                try:
                    response_text = await asyncio.wait_for(
                        asyncio.shield(response_task),
                        timeout=FOREGROUND_STANDBY_AFTER_S,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "Gateway voice turn exceeded {}s; keeping request alive and sending standby.",
                        FOREGROUND_STANDBY_AFTER_S,
                    )
                    foreground_standby_started = True
                    await websocket.send_json({"type": "background_task_started"})
                    await emit_filtered("Working on it, I'll respond when finished.")
                    try:
                        response_text = await asyncio.wait_for(
                            asyncio.shield(response_task),
                            timeout=BACKGROUND_RESEARCH_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Gateway voice turn exceeded additional {}s; keeping pending state and waiting for any later session final.",
                            BACKGROUND_RESEARCH_TIMEOUT_S,
                        )
                        response_text = ""
                raw_stream_chars = len(response_text)
                # Re-resolve the session JSONL path: when this is the first
                # turn on a fresh WS, sessions.json may not have had the
                # entry yet at snapshot time. The orchestrator creates the
                # session file during the turn, so it's available now.
                if session_path is None:
                    session_path = await get_session_file_path(
                        session_owner_key
                    )
                final_events = await read_assistant_final_events_after(
                    session_path, session_offset
                )
                if final_events:
                    delivered_session_message_ids.update(
                        str(event.get("id") or "")
                        for event in final_events
                        if str(event.get("id") or "").strip()
                    )
                    latest_offset = (
                        session_path.stat().st_size
                        if session_path and session_path.exists()
                        else session_offset
                    )
                    await persist_delivery_state(session_offset=latest_offset)
                spoken: Optional[str] = (
                    str(final_events[-1].get("text") or "").strip()
                    if final_events
                    else None
                )

                if spoken:
                    if foreground_standby_started:
                        await websocket.send_json({"type": "background_task_finished"})
                    await emit_filtered(clean_for_display(spoken))
                elif is_placeholder_gateway_response(response_text):
                    logger.info(
                        "Gateway returned placeholder response with no final yet; keeping pending state."
                    )
                    await websocket.send_json({"type": "background_task_started"})
                elif response_text and not looks_like_reasoning_leak(response_text):
                    # No <final> block was found in the JSONL (either the
                    # orchestrator chose not to emit one, or we couldn't
                    # reach the file) but the response body itself doesn't
                    # look like leaked reasoning. Emit it directly with
                    # clean_for_display, which strips any stray <think>
                    # blocks and unwraps <final> tags without dropping
                    # legitimate plain-prose answers. Do NOT route through
                    # the strict-final sanitizer here — strict mode would
                    # drop everything when no <final> tag is present and
                    # produce silent dead-air.
                    if foreground_standby_started:
                        await websocket.send_json({"type": "background_task_finished"})
                    await emit_filtered(clean_for_display(response_text))
                else:
                    # Either the response was empty, or the gateway handed
                    # us tag-stripped reasoning prose. Refuse to speak it
                    # and keep any pending UI state instead of leaking.
                    if response_text:
                        logger.warning(
                            "Gateway returned {} chars with no <final> block "
                            "and reasoning-leak markers; suppressing spoken fallback.",
                            raw_stream_chars,
                        )
            else:
                async for chunk in backend.chat_stream(transcript, user_key=session_id):
                    raw_stream_chars += len(chunk)
                    await emit_filtered(sanitizer.feed(chunk))

                await emit_filtered(sanitizer.flush())

        # Flush remaining text as the last audio segment.
        leftover = sentence_buffer.strip()
        if leftover:
            await tts_queue.put(leftover)

        copy_candidate = clean_for_display(full_response).strip()
        if copy_candidate and not wants_email_copy:
            last_spoken_answer = copy_candidate
            _, fg_offset = await snapshot_session_offset(session_owner_key)
            await persist_delivery_state(session_offset=fg_offset)
            confirmation = await maybe_send_pending_email_copy(
                copy_candidate,
                source="foreground",
            )
            if confirmation:
                await emit_filtered(confirmation)

        # Signal the worker to finish, then wait for all audio to be sent.
        await tts_queue.put(None)
        await tts_task

        # Ensure we wait a tiny bit for the physical player buffer to drain before finishing.
        await asyncio.sleep(0.5)

        await websocket.send_json({
            "type": "response_complete",
            "text": full_response,
        })
        logger.info(f"Response complete: {full_response[:100]}...")

    async def deliver_task_result(
        result_text: str,
        *,
        source_offset: Optional[int] = None,
        max_chars: int = VOICE_RESULT_NORMAL_MAX_CHARS,
        max_sentences: int = VOICE_RESULT_NORMAL_MAX_SENTENCES,
    ):
        """Deliver a completed background-task result directly to the client."""
        nonlocal last_spoken_answer
        display_text = cap_voice_result(
            result_text,
            max_chars=max_chars,
            max_sentences=max_sentences,
        )
        if not display_text:
            display_text = clean_for_display(result_text).strip() or result_text.strip()
        if not display_text:
            return
        last_spoken_answer = display_text
        await persist_delivery_state(session_offset=source_offset)
        email_confirmation = await maybe_send_pending_email_copy(
            display_text,
            source="delayed",
            source_offset=source_offset,
        )
        if email_confirmation:
            display_text = f"{display_text} {email_confirmation}"

        await websocket.send_json({"type": "background_task_finished"})
        await websocket.send_json({"type": "assistant_turn_start"})
        await websocket.send_json({
            "type": "response_chunk",
            "text": display_text,
        })

        seq = 0
        for chunk in split_text_for_tts(display_text):
            aac_bytes = await tts.synthesize_aac(chunk)
            if not aac_bytes:
                continue
            audio_b64 = base64.b64encode(aac_bytes).decode()
            await websocket.send_json({
                "type": "audio_aac",
                "data": audio_b64,
                "seq": seq,
                "mime": getattr(tts, "mime_type", "audio/aac"),
            })
            seq += 1

        await asyncio.sleep(0.5)
        await websocket.send_json({
            "type": "response_complete",
            "text": display_text,
        })
        logger.info(f"Delivered task result: {display_text[:100]}...")

    async def watch_session_announces():
        """Deliver new background-task completion announcements for this session."""
        nonlocal active_task, pending_email_copy_request, last_spoken_answer

        state = await read_voice_delivery_state(session_owner_key)
        delivered_announce_task_ids.update(
            str(task_id)
            for task_id in (state.get("taskIds") or [])
            if str(task_id).strip()
        )
        delivered_session_message_ids.update(
            str(message_id)
            for message_id in (state.get("messageIds") or [])
            if str(message_id).strip()
        )
        pending_email_copy_request = _normalize_pending_email_copy(
            state.get("pendingEmailCopy")
        )
        if state.get("lastSpokenAnswer"):
            last_spoken_answer = state["lastSpokenAnswer"]

        existing_tasks = await get_tasks()
        if not delivered_announce_task_ids:
            delivered_announce_task_ids.update(
                task.get("taskId") or task.get("id")
                for task in existing_tasks
                if is_session_announce_task(task)
            )
        session_file_path = await get_session_file_path(session_owner_key)
        saved_offset = state.get("sessionOffset")
        if isinstance(saved_offset, int):
            session_file_offset = saved_offset
        else:
            session_file_offset = (
                session_file_path.stat().st_size
                if session_file_path and session_file_path.exists()
                else 0
            )
        await persist_delivery_state(session_offset=session_file_offset)
        logger.info(
            f"Announce watcher ready for {session_owner_key} "
            f"with {len(delivered_announce_task_ids)} existing completion task(s), "
            f"{len(delivered_session_message_ids)} delivered session message(s); "
            f"session_file={session_file_path or 'unresolved'}"
        )

        while True:
            await asyncio.sleep(ANNOUNCE_POLL_INTERVAL_S)

            if is_listening or (active_task and not active_task.done()):
                continue

            delivery_failed = False
            delivered_outbox_ids: set[str] = set()
            outbox_items = await load_voice_result_outbox(session_owner_key)
            for item in outbox_items:
                item_id = str(item.get("id") or "").strip()
                item_text = str(item.get("text") or "").strip()
                item_max_chars = int(item.get("maxChars") or VOICE_RESULT_NORMAL_MAX_CHARS)
                item_max_sentences = int(item.get("maxSentences") or VOICE_RESULT_NORMAL_MAX_SENTENCES)
                if not item_id or item_id in delivered_session_message_ids:
                    delivered_outbox_ids.add(item_id)
                    continue
                active_task = asyncio.create_task(
                    deliver_task_result(
                        item_text,
                        source_offset=None,
                        max_chars=item_max_chars,
                        max_sentences=item_max_sentences,
                    )
                )
                try:
                    await active_task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Voice outbox delivery failed for {item_id}: {e}")
                    delivery_failed = True
                    break
                delivered_session_message_ids.add(item_id)
                delivered_outbox_ids.add(item_id)

            if delivery_failed:
                continue

            tasks = await get_tasks()
            new_announces = []
            for task in tasks:
                if not is_session_announce_task(task):
                    continue
                task_id = task.get("taskId") or task.get("id")
                if not task_id or task_id in delivered_announce_task_ids:
                    continue
                new_announces.append(task)

            if not session_file_path:
                session_file_path = await get_session_file_path(session_owner_key)
                session_file_offset = (
                    session_file_path.stat().st_size
                    if session_file_path and session_file_path.exists()
                    else 0
                )

            session_events: list[dict] = []
            if session_file_path:
                next_session_file_offset, session_events = await read_new_session_events(
                    session_file_path,
                    session_file_offset,
                )
            else:
                next_session_file_offset = session_file_offset

            if not new_announces and not session_events and not delivered_outbox_ids:
                continue

            new_announces.sort(key=lambda task: task.get("createdAt") or task.get("endedAt") or "")
            for task in new_announces:
                task_id = task.get("taskId") or task.get("id")
                payload = (task.get("task") or "").strip()
                if not payload:
                    logger.debug(f"Skipping empty announce payload for task {task_id}")
                    delivered_announce_task_ids.add(task_id)
                    continue

                logger.info(f"Delivering announce task {task_id} to voice session")
                result_text = extract_task_result_text(payload)
                if is_nonfinal_background_result(result_text):
                    logger.info(
                        f"Suppressing non-final announce task {task_id}: {result_text[:100]}"
                    )
                    delivered_announce_task_ids.add(task_id)
                    continue
                active_task = asyncio.create_task(
                    deliver_task_result(
                        result_text,
                        source_offset=None,
                        max_chars=voice_result_max_chars,
                        max_sentences=voice_result_max_sentences,
                    )
                )
                try:
                    await active_task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Announce delivery failed for task {task_id}: {e}")
                    delivery_failed = True
                    break
                delivered_announce_task_ids.add(task_id)

            if delivery_failed:
                continue

            for event in session_events:
                event_id = str(event.get("id") or "").strip()
                event_kind = str(event.get("kind") or "").strip()
                if not event_id or event_id in delivered_session_message_ids:
                    continue
                payload = str(event.get("text") or "").strip()
                if not payload:
                    continue
                if event_kind == "announce":
                    # In background-research voice mode, subagent announces
                    # are intermediate data for the parent session, not user-
                    # facing speech. They are often markdown-heavy or partial.
                    # Wait for the parent assistant_final aggregate instead.
                    logger.info(f"Suppressing raw session announce {event_id}")
                    delivered_session_message_ids.add(event_id)
                    continue
                logger.info(f"Delivering session {event_kind} {event_id} to voice session")
                result_text = (
                    extract_task_result_text(payload)
                    if event_kind == "announce"
                    else payload
                )
                if is_nonfinal_background_result(result_text):
                    logger.info(
                        f"Suppressing non-final session {event_kind} {event_id}: {result_text[:100]}"
                    )
                    delivered_session_message_ids.add(event_id)
                    continue
                active_task = asyncio.create_task(
                    deliver_task_result(
                        result_text,
                        source_offset=next_session_file_offset,
                        max_chars=voice_result_max_chars,
                        max_sentences=voice_result_max_sentences,
                    )
                )
                try:
                    await active_task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Session event delivery failed for {event_id}: {e}")
                    delivery_failed = True
                    break
                delivered_session_message_ids.add(event_id)

            if delivery_failed:
                continue

            session_file_offset = next_session_file_offset
            await remove_voice_result_outbox_entries(session_owner_key, delivered_outbox_ids)
            await persist_delivery_state(session_offset=session_file_offset)

    # Run process_transcript as a background task so the receive loop
    # stays responsive to pings (prevents timeout during long AI calls).
    active_task: asyncio.Task | None = None
    announce_task = asyncio.create_task(watch_session_announces())

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "start_listening":
                is_listening = True
                audio_buffer = []
                await websocket.send_json({"type": "listening_started"})
                logger.debug("Started listening")

            elif msg["type"] == "stop_listening":
                is_listening = False

                if audio_buffer:
                    audio_data = np.concatenate(audio_buffer)
                    logger.debug("Transcribing audio...")
                    transcript = await stt.transcribe(audio_data)
                    if active_task and not active_task.done():
                        active_task.cancel()
                    active_task = asyncio.create_task(process_transcript(transcript))

                audio_buffer = []
                await websocket.send_json({"type": "listening_stopped"})
                logger.debug("Stopped listening")

            elif msg["type"] == "text_input":
                transcript = (msg.get("text") or "").strip()
                if active_task and not active_task.done():
                    active_task.cancel()
                active_task = asyncio.create_task(
                    process_transcript(transcript, echo_transcript=False)
                )

            elif msg["type"] == "audio" and is_listening:
                audio_bytes = base64.b64decode(msg["data"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_buffer.append(audio_np)

                if vad and len(audio_np) > 0:
                    has_speech = vad.is_speech(audio_np)
                    await websocket.send_json({
                        "type": "vad_status",
                        "speech_detected": has_speech,
                    })

            elif msg["type"] == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if announce_task and not announce_task.done():
            announce_task.cancel()
        if active_task and not active_task.done():
            active_task.cancel()


# Serve static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
