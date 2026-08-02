"""
Tests for speech/display text cleanup helpers and the streaming sanitizer.

The streaming-sanitizer block guards against the May 2026 voice-chat
thinking-leak bug, where the OpenClaw gateway's OpenAI-compat stream emitted
Gemini 3 Flash's native thinking content as plain delta.content alongside the
tagged <final>...</final> reply.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server.text_utils import (
    StreamSanitizer,
    clean_for_display,
    clean_for_speech,
    extract_last_final,
    is_silent_control_response,
    looks_like_reasoning_leak,
    strip_final_wrappers,
    strip_thinking,
)

# ---------------------------------------------------------------------------
# clean_for_speech (existing real-estate / address coverage, restored)
# ---------------------------------------------------------------------------


def test_clean_for_speech_expands_real_estate_shorthand():
    cleaned = clean_for_speech("$1.5M, 3bd/2ba, 1,410 sq ft on Harbor Rd.")

    assert "one point five million dollars" in cleaned
    assert "three bedroom, two bath" in cleaned
    assert "one thousand four hundred ten square feet" in cleaned
    assert "Harbor Road" in cleaned


def test_clean_for_speech_expands_local_street_suffixes():
    cleaned = clean_for_speech("229 Harbor View Blvd, Lakeside and 327 Riverview Ave")

    assert "Harbor View Boulevard" in cleaned
    assert "Riverview Avenue" in cleaned


def test_clean_for_speech_handles_plain_dollars_and_hoa():
    cleaned = clean_for_speech("$685,000 condo with 2bd/1ba and $250 HOA")

    assert "six hundred eighty five thousand dollars" in cleaned
    assert "two bedroom, one bath" in cleaned
    assert "two hundred fifty dollars H O A" in cleaned


def test_clean_for_speech_handles_decimal_baths():
    cleaned = clean_for_speech("3bd/2.5ba on Encino Dr")

    assert "three bedroom, two and a half bath" in cleaned
    assert "Encino Drive" in cleaned


def test_clean_for_speech_handles_common_mls_acronyms():
    cleaned = clean_for_speech("SFR with W/D, WIC, FP, A/C and $/Sqft noted in MLS")

    assert "single family residence" in cleaned
    assert "washer dryer" in cleaned
    assert "walk-in closet" in cleaned
    assert "fireplace" in cleaned
    assert "air conditioning" in cleaned
    assert "dollars per square foot" in cleaned
    assert "M L S" in cleaned


def test_clean_for_speech_expands_street_suffix_only_in_address_context():
    cleaned = clean_for_speech("105 Wesley St near Lakeside, but Dr. Smith is the broker")

    assert "Wesley Street" in cleaned
    assert "Dr. Smith" in cleaned


def test_clean_for_speech_expands_state_abbreviation_in_address_context():
    cleaned = clean_for_speech("327 Riverview Ave, Lakeside, CA 95010")

    assert "Riverview Avenue" in cleaned
    assert "California 95010" in cleaned


def test_clean_for_speech_does_not_expand_non_address_ca():
    cleaned = clean_for_speech("Use CA glue in the garage.")

    assert "CA glue" in cleaned


# ---------------------------------------------------------------------------
# strip_thinking / strip_final_wrappers
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_block():
    text = "<think>internal reasoning</think>Hello there."
    assert strip_thinking(text) == "Hello there."


def test_strip_thinking_handles_attributes_and_case():
    text = '<Think foo="bar">stuff</THINK>visible'
    assert strip_thinking(text) == "visible"


def test_strip_final_wrappers_keeps_inner_text():
    text = "<final>The capital of France is Paris.</final>"
    assert strip_final_wrappers(text) == "The capital of France is Paris."


def test_clean_for_display_strips_think_and_final_tags():
    raw = "<think>plan</think><final>Hello!</final>"
    assert clean_for_display(raw) == "Hello!"


def test_clean_for_speech_strips_think_and_final_tags():
    raw = "<think>plan</think><final>Hello, friend.</final>"
    assert clean_for_speech(raw).strip() == "Hello, friend."


@pytest.mark.parametrize(
    "text",
    [
        "NO_REPLY",
        " no reply ",
        "No-Reply.",
        "`NO_REPLY`",
        "<final>NO_REPLY</final>",
        "<think>stay silent</think><final>NO_REPLY</final>",
    ],
)
def test_silent_control_response_is_suppressed(text):
    assert is_silent_control_response(text)
    assert clean_for_display(text) == ""
    assert clean_for_speech(text) == ""


def test_silent_control_response_does_not_hide_normal_prose():
    text = "The token NO_REPLY should not be spoken by the voice layer."
    assert not is_silent_control_response(text)
    assert clean_for_display(text) == text


# ---------------------------------------------------------------------------
# StreamSanitizer — strict mode (openclaw:* gateway)
# ---------------------------------------------------------------------------


def _drive(sanitizer: StreamSanitizer, chunks: list[str]) -> str:
    parts = [sanitizer.feed(c) for c in chunks]
    parts.append(sanitizer.flush())
    return "".join(parts)


def test_strict_emits_only_inner_final():
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<final>Hello world.</final>"])
    assert out == "Hello world."


def test_strict_drops_leading_thinking_prose():
    """The exact shape of the screenshot leak: prose before <final>."""
    s = StreamSanitizer(strict_final=True)
    leaked = (
        "much useful.\nBut wait, the cron list showed: real-estate-scout. "
        "I'll just spawn the sub-agent.\n\nVoice style: 1-2 short sentences.\n"
    )
    out = _drive(
        s,
        [
            leaked,
            "<final>I'll get Claude Sonnet on it right away.</final>",
        ],
    )
    assert out == "I'll get Claude Sonnet on it right away."


def test_strict_drops_trailing_tool_call_noise():
    s = StreamSanitizer(strict_final=True)
    out = _drive(
        s,
        [
            "<final>Done!</final>",
            ' tool_call={"name":"foo"}',
        ],
    )
    assert out == "Done!"


def test_strict_handles_split_open_tag_across_chunks():
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<fin", "al>Hi there.</final>"])
    assert out == "Hi there."


def test_strict_handles_split_close_tag_across_chunks():
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<final>Hi there.</fin", "al>"])
    assert out == "Hi there."


def test_strict_handles_intra_word_chunk_split_inside_final():
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<final>Hi ", "there", " friend.</final>"])
    assert out == "Hi there friend."


def test_strict_drops_buggy_unclosed_open_tag_fragment():
    """Reproduces the exact gateway streaming bug observed live: only `<final` arrives."""
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<final"])
    assert out == ""


def test_strict_emits_buffered_content_when_close_tag_truncated():
    """If the gateway eats the closing </final>, we still surface the inner text."""
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<final>The answer is 42."])
    assert out == "The answer is 42."


def test_strict_strips_thinking_tags_too():
    s = StreamSanitizer(strict_final=True)
    out = _drive(s, ["<think>rambling</think><final>Crisp answer.</final>"])
    assert out == "Crisp answer."


def test_strict_drops_thinking_outside_final_even_after_final_seen():
    s = StreamSanitizer(strict_final=True)
    out = _drive(
        s,
        [
            "<final>Answer A.</final>",
            "<think>more reasoning</think>",
            "<final>Answer B.</final>",
        ],
    )
    assert out == "Answer A.Answer B."


# ---------------------------------------------------------------------------
# StreamSanitizer — lenient mode (direct OpenAI / non-orchestrator)
# ---------------------------------------------------------------------------


def test_lenient_passes_plain_content_through():
    s = StreamSanitizer(strict_final=False)
    out = _drive(s, ["Hello, world."])
    assert out == "Hello, world."


def test_lenient_strips_thinking_keeps_rest():
    s = StreamSanitizer(strict_final=False)
    out = _drive(s, ["<think>plan</think>Visible answer."])
    assert out == "Visible answer."


def test_lenient_handles_split_think_close_tag():
    s = StreamSanitizer(strict_final=False)
    out = _drive(s, ["<think>plan</thi", "nk>visible"])
    assert out == "visible"


def test_lenient_still_unwraps_final_tags_when_seen():
    """Lenient mode passes through plain prose but, when <final>...</final>
    happens to appear, still unwraps it (consistent with the orchestrator
    contract). Anything BEFORE the opener is preserved (unlike strict mode)."""
    s = StreamSanitizer(strict_final=False)
    out = _drive(s, ["preamble <final>inner</final> trailer"])
    assert out == "preamble inner trailer"


def test_lenient_drops_partial_tag_at_eof_to_avoid_leaking_fragment():
    s = StreamSanitizer(strict_final=False)
    out = _drive(s, ["abc<thi"])
    assert out == "abc"


# ---------------------------------------------------------------------------
# Replays the exact captured bug (session 67a4ef71, message 91f016ad)
# ---------------------------------------------------------------------------


def test_strict_recovers_real_world_leaked_screenshot_payload():
    """Concatenation of the model's actual content blocks for the May 7 voice
    request that produced the user's screenshot. With strict-final filtering,
    only the inner <final> text should reach TTS / display."""
    s = StreamSanitizer(strict_final=True)
    leaked_payload = (
        'The user wants to fix the "real estate clone job". Looking at the '
        "cron history, the `real-estate-scout` job recently failed with a "
        'timeout, and there have been previous "no space left on device" '
        "errors. \n\nWait, I should check if there's a file specifically called "
        '"clone" or related in `tools/`. \nThe `ls -R | grep -i clone` '
        "didn't show much useful.\nBut wait, the `cron list` showed:\n"
        "`real-estate-scout` - id `fb8ce8fc-f420-4062-ae53-b58ac5255027`.\n\n"
        'Actually, the user said "fix the real estate clone job". \n'
        "I'll just spawn the sub-agent with that exact task description and "
        "let it investigate.\n\nVoice style: 1-2 short sentences, no markdown.\n"
        "\"I'll get Claude Sonnet on it right away. I'm spawning a sub-agent "
        'now to investigate and fix the real estate job."'
        "<final>I'll get Claude Sonnet on it right away. I'm spawning a "
        "sub-agent now to investigate and fix the real estate job.</final>"
    )
    out = _drive(s, [leaked_payload])
    assert out == (
        "I'll get Claude Sonnet on it right away. I'm spawning a "
        "sub-agent now to investigate and fix the real estate job."
    )


# ---------------------------------------------------------------------------
# Stream-level invariant: chunking must not change the result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "<think>plan</think><final>Hi.</final>",
        "leak prose<final>Hi friend.</final>",
        "<final>Multi-sentence answer. With two parts.</final>",
        "<final>Unterminated answer",
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 13, 64])
def test_strict_is_chunk_invariant(payload: str, chunk_size: int):
    chunks = [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]
    full_at_once = _drive(StreamSanitizer(strict_final=True), [payload])
    chunked = _drive(StreamSanitizer(strict_final=True), chunks)
    assert chunked == full_at_once


# ---------------------------------------------------------------------------
# extract_last_final / looks_like_reasoning_leak
#
# Backstop for the May 8 leak: the gateway's non-streaming OpenAI-compat path
# strips <think>/<final> markers and concatenates reasoning + answer as flat
# prose, which slips past the streaming sanitizer's tag-presence heuristic.
# The voice server now reads the authoritative <final> from the orchestrator
# session JSONL and uses ``looks_like_reasoning_leak`` as a defensive guard.
# ---------------------------------------------------------------------------


def test_extract_last_final_returns_inner_text():
    text = (
        "<think>weighing options</think>"
        "<final>I see your appointment at 8:30, "
        "but I do not have a location saved.</final>"
    )
    assert extract_last_final(text) == (
        "I see your appointment at 8:30, but I do not have a location saved."
    )


def test_extract_last_final_picks_last_block_when_multiple():
    text = (
        "<final>first attempt</final>"
        " trailing reasoning "
        "<final>second corrected answer</final>"
    )
    assert extract_last_final(text) == "second corrected answer"


def test_extract_last_final_returns_none_when_absent():
    assert extract_last_final("plain prose with no markers") is None
    assert extract_last_final("") is None


def test_extract_last_final_handles_unclosed_block_as_none():
    """Defensive: an unclosed <final> at end-of-message is not authoritative."""
    assert extract_last_final("<final>partial answer with no closer") is None


def test_extract_last_final_strips_whitespace():
    text = "<final>\n  hello there\n</final>"
    assert extract_last_final(text) == "hello there"


def test_looks_like_reasoning_leak_catches_first_person_planning():
    leak = (
        "The user is asking for the next best route in voice mode. "
        "I need to check the calendar. Plan: 1. Look up the calendar list. "
        "2. If a location is found, give the route."
    )
    assert looks_like_reasoning_leak(leak) is True


def test_looks_like_reasoning_leak_catches_real_screenshot_payload():
    """The exact pattern from the May 8 voice screenshot the user reported."""
    leak = (
        'The user is asking "What is the next best route" in voice mode.\n'
        "According to the prompt instructions:\n"
        '- "What is the next best route" usually refers to a regular route.\n'
        'Wait, the prompt says "For calendar/reminder questions, check '
        'calendar/reminders only".\n'
        "Let's check the calendar for the next event.\n"
        "I'll check the calendar list."
    )
    assert looks_like_reasoning_leak(leak) is True


def test_looks_like_reasoning_leak_passes_short_clean_answer():
    assert looks_like_reasoning_leak("Hello! How can I help you today?") is False


def test_looks_like_reasoning_leak_passes_clean_commute_answer():
    answer = "The Google Maps estimate is twenty-one minutes right now " "with light traffic."
    assert looks_like_reasoning_leak(answer) is False


def test_looks_like_reasoning_leak_passes_clean_calendar_answer():
    answer = (
        "I see your appointment scheduled for 8:30, but I do not have a "
        "location saved for it in your calendar."
    )
    assert looks_like_reasoning_leak(answer) is False


def test_looks_like_reasoning_leak_handles_empty_input():
    assert looks_like_reasoning_leak("") is False
    assert looks_like_reasoning_leak("   ") is False


def test_looks_like_reasoning_leak_passes_market_research_answer():
    """Gateway returned a clean two-sentence market answer
    with no <final> tags. The voice server must classify this as NOT a leak
    so it gets spoken instead of suppressed as dead-air."""
    answer = (
        "Analysts remain mostly optimistic despite recent volatility, "
        "pointing to steady adoption and a strong product roadmap. "
        "There is some negative chatter regarding execution risk and "
        "concerns that competitive tools might squeeze margins in the "
        "coming months."
    )
    assert looks_like_reasoning_leak(answer) is False


def test_clean_for_display_passes_flat_prose_answer_unchanged():
    """If the gateway returns a plain-prose answer (tags already stripped),
    clean_for_display must not eat it. This is the fallback path the voice
    server uses when the session JSONL has no <final> block."""
    answer = (
        "Analysts remain mostly optimistic despite the recent lack of "
        "performance, pointing to strong AI adoption."
    )
    cleaned = clean_for_display(answer)
    assert "Analysts remain mostly optimistic" in cleaned
    assert "AI adoption" in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
