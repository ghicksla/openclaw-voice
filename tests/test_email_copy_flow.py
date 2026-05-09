"""
Unit tests for durable voice email-copy handling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server.main import (
    _can_fulfill_pending_email_copy,
    _make_pending_email_copy_request,
    _normalize_pending_email_copy,
    _pick_recent_copy_candidate,
    detect_voice_intents,
    is_send_copy_to_email_request,
)


def test_send_copy_intent_matches_explicit_phrase():
    assert is_send_copy_to_email_request("That's great send a copy of that to my email")
    assert is_send_copy_to_email_request("Please email that to my email")
    assert is_send_copy_to_email_request("Splendid please email it to my Gmail")


def test_voice_intent_router_matches_email_copy_variants():
    assert "email_copy" in detect_voice_intents("Please email this to my inbox")
    assert "email_copy" in detect_voice_intents("Send that response to my email")
    assert "email_copy" in detect_voice_intents("Could you forward it to my Gmail?")


def test_send_copy_intent_rejects_non_email_prompt():
    assert not is_send_copy_to_email_request("Can you summarize that")
    assert not is_send_copy_to_email_request("What is the product outlook this year")


def test_pick_recent_copy_candidate_skips_status_text():
    texts = [
        "I do not have a finished answer to copy yet. I queued it and will send it as soon as that answer is ready.",
        "Working on it, I'll respond when finished.",
        "The company outlook is strong and adoption odds remain high.",
    ]
    assert _pick_recent_copy_candidate(texts) == "The company outlook is strong and adoption odds remain high."


def test_normalize_pending_email_copy_requires_recipient():
    assert _normalize_pending_email_copy(None) is None
    assert _normalize_pending_email_copy({}) is None
    assert _normalize_pending_email_copy({"toAddress": ""}) is None


def test_pending_copy_request_requires_delayed_source_when_configured():
    pending = _make_pending_email_copy_request(
        to_address="user@example.com",
        min_session_offset=100,
        require_delayed=True,
    )
    assert not _can_fulfill_pending_email_copy(
        pending,
        source="foreground",
        source_offset=120,
        text="Here is the summary.",
    )
    assert _can_fulfill_pending_email_copy(
        pending,
        source="delayed",
        source_offset=120,
        text="Here is the summary.",
    )


def test_pending_copy_request_enforces_min_session_offset():
    pending = _make_pending_email_copy_request(
        to_address="user@example.com",
        min_session_offset=250,
        require_delayed=False,
    )
    assert not _can_fulfill_pending_email_copy(
        pending,
        source="delayed",
        source_offset=200,
        text="Here is the summary.",
    )
    assert _can_fulfill_pending_email_copy(
        pending,
        source="delayed",
        source_offset=300,
        text="Here is the summary.",
    )
