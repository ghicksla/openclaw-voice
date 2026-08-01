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
    build_email_copy_body,
    detect_voice_intents,
    extract_compound_email_task_text,
    is_send_copy_to_email_request,
    should_delay_email_copy_request,
)


def test_send_copy_intent_matches_explicit_phrase():
    assert is_send_copy_to_email_request("That's great send a copy of that to my email")
    assert is_send_copy_to_email_request("Please email that to my email")
    assert is_send_copy_to_email_request("Splendid please email it to my Gmail")


def test_voice_intent_router_matches_email_copy_variants():
    assert "email_copy" in detect_voice_intents("Please email this to my inbox")
    assert "email_copy" in detect_voice_intents("Send that response to my email")
    assert "email_copy" in detect_voice_intents("Could you forward it to my Gmail?")


def test_delay_email_copy_request_waits_for_background_result():
    assert should_delay_email_copy_request("Email it to me when done please")
    assert should_delay_email_copy_request("Send this to my inbox", has_background_work=True)
    assert not should_delay_email_copy_request("Send this to my inbox")


def test_extract_compound_email_task_text_keeps_task_part():
    task = extract_compound_email_task_text(
        "What are the top 10 movies currently playing and send me an email when done"
    )
    assert task == "What are the top 10 movies currently playing"
    assert extract_compound_email_task_text("Please email this to my inbox") is None


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


def test_email_copy_body_does_not_number_single_summary():
    body = build_email_copy_body(["The company outlook is improving."])
    assert "1. The company outlook is improving." not in body
    assert "The company outlook is improving." in body


def test_email_copy_body_numbers_multiple_summaries():
    body = build_email_copy_body(["First summary.", "Second summary."])
    assert "1. First summary." in body
    assert "2. Second summary." in body


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
