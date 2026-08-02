"""
Text utilities for voice-friendly output.

Cleans AI responses for TTS and display — removes markdown, thinking
tags, tables, prompt leakage, etc.
"""

import re
from typing import Optional

_THINK_OPEN_RE = re.compile(r"<\s*think\b[^<>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_FINAL_OPEN_RE = re.compile(r"<\s*final\b[^<>]*>", re.IGNORECASE)
_FINAL_CLOSE_RE = re.compile(r"<\s*/\s*final\s*>", re.IGNORECASE)
_FINAL_BLOCK_RE = re.compile(
    r"<\s*final\b[^<>]*>([\s\S]*?)<\s*/\s*final\s*>",
    re.IGNORECASE,
)


def extract_last_final(text: str) -> Optional[str]:
    """Return the text inside the last ``<final>...</final>`` block, if any.

    The OpenClaw orchestrator's tagged-output contract uses this block as the
    only user-visible portion of a turn. The voice server reads this from the
    session JSONL because the gateway's OpenAI-compat shim strips the tags
    before returning the response body, which would otherwise turn the
    reasoning into spoken output.
    """
    if not text:
        return None
    matches = _FINAL_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


_LEAK_INDICATORS = (
    "the user is asking",
    "the user wants",
    "i need to ",
    "i'll check",
    "i'll try",
    "i should ",
    "let me check",
    "let's check",
    "let me see",
    "wait, ",
    "actually,",
    "plan:",
    "constraint:",
    "constraints:",
    "instruction:",
    "instructions:",
    "voice mode hard limit",
    "voice mode constraint",
    "voice mode requirement",
    "according to the prompt",
    "tool call",
    "tool: ",
    "first, i",
    "next, i",
    "finally, i",
)


def looks_like_reasoning_leak(text: str) -> bool:
    """Heuristic: detect first-person planning prose meant for ``<think>``.

    Used as a defensive guard when the upstream gateway has stripped the
    ``<think>``/``<final>`` markers and we're trying to decide whether the
    flat response body is the user-visible answer or a leaked reasoning blob.
    """
    if not text:
        return False
    sample = text.strip().lower()
    if len(sample) > 600:
        sample = sample[:600]
    matches = sum(1 for marker in _LEAK_INDICATORS if marker in sample)
    if matches >= 2:
        return True
    if len(text) >= 400 and matches >= 1:
        return True
    return False


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks (Gemini/DeepSeek/orchestrator thinking tags)."""
    return re.sub(
        r"<\s*think\b[^<>]*>[\s\S]*?<\s*/\s*think\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )


def strip_final_wrappers(text: str) -> str:
    """
    Strip the <final> and </final> tag tokens themselves while keeping the
    content between them. Used as defense in depth for messages that survived
    the streaming sanitizer with literal tag fragments embedded.
    """
    text = _FINAL_OPEN_RE.sub("", text)
    text = _FINAL_CLOSE_RE.sub("", text)
    return text


def is_silent_control_response(text: str) -> bool:
    """Return True for OpenClaw's exact no-response control sentinel.

    This is deliberately an exact normalized match so ordinary prose that
    happens to mention ``NO_REPLY`` is still displayed and spoken.
    """
    if not text:
        return False
    normalized = strip_final_wrappers(strip_thinking(text)).strip()
    return bool(
        re.fullmatch(
            r"""[`*_'"]*NO[\s_-]*REPLY[`*_'"]*[\s.!?]*""",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def strip_tables(text: str) -> str:
    """Remove markdown table rows (lines dominated by pipes and dashes)."""
    lines = text.split("\n")
    cleaned = [line for line in lines if not re.match(r"^\s*\|", line)]
    return "\n".join(cleaned)


def clean_for_display(text: str) -> str:
    """Clean response text for on-screen display (lighter than TTS cleaning)."""
    if not text:
        return text
    if is_silent_control_response(text):
        return ""
    text = strip_thinking(text)
    text = strip_final_wrappers(text)
    text = strip_tables(text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def clean_for_speech(text: str) -> str:
    """
    Clean text for TTS rendering.

    Removes markdown, thinking tags, tables, URLs, emojis, etc.
    """
    if not text:
        return text
    if is_silent_control_response(text):
        return ""

    text = strip_thinking(text)
    text = strip_final_wrappers(text)
    text = strip_tables(text)

    # Remove code blocks (``` … ```)
    text = re.sub(r"```[\s\S]*?```", " code block omitted ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Hashtags → keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # URLs and markdown links
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Technical emojis
    text = re.sub(r"[🔗📦📁💻🖥️⚡🔧🛠️📝✅❌⚠️🚀🎯💡🔍📊📈📉🗂️📋]", "", text)

    # Bullet points
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    # Colons before newlines → period
    text = re.sub(r":\s*\n", ". ", text)

    # Real-estate/domain cleanup before provider TTS.
    text = _normalize_currency_for_speech(text)
    text = _normalize_listing_acronyms_for_speech(text)
    text = _normalize_real_estate_shorthand_for_speech(text)
    text = _normalize_measurements_for_speech(text)
    text = _normalize_address_suffixes_for_speech(text)
    text = _normalize_state_abbreviations_for_speech(text)

    # Convert machine-formatted time expressions to natural speech.
    text = _normalize_times_for_speech(text)

    # Collapse whitespace
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    text = re.sub(r"\.(\s*\.)+", ".", text)

    return text


def _normalize_currency_for_speech(text: str) -> str:
    money_re = re.compile(r"\$(\d[\d,]*)(?:\.(\d+))?(?:\s*([KMBkmb]))?\b")
    scale_words = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
    }

    def _money_sub(m: re.Match[str]) -> str:
        whole = m.group(1).replace(",", "")
        frac = (m.group(2) or "").rstrip("0")
        scale = (m.group(3) or "").lower()

        if scale:
            number_token = whole if not frac else f"{whole}.{frac}"
            return f"{_number_token_to_words(number_token)} {scale_words[scale]} dollars"

        dollars = int(whole or "0")
        if frac:
            cents = int((frac + "00")[:2])
            if cents:
                dollar_words = _number_token_to_words(str(dollars))
                cent_words = _number_token_to_words(str(cents))
                dollar_label = "dollar" if dollars == 1 else "dollars"
                cent_label = "cent" if cents == 1 else "cents"
                return f"{dollar_words} {dollar_label} and {cent_words} {cent_label}"

        words = _number_token_to_words(str(dollars))
        label = "dollar" if dollars == 1 else "dollars"
        return f"{words} {label}"

    return money_re.sub(_money_sub, text)


def _normalize_listing_acronyms_for_speech(text: str) -> str:
    # Common MLS shorthand from CRMLS-style listings.
    substitutions = (
        (r"\bBr\s*/\s*Ba\b", "bedroom bath"),
        (r"\$\s*/\s*Sqft\b", "dollars per square foot"),
        (r"\bYrBuilt\b", "year built"),
        (r"\bSFR\b", "single family residence"),
        (r"\bTWNHS\b", "townhouse"),
        (r"\bCOOP\b", "co-op"),
        (r"\bDPLX\b", "duplex"),
        (r"\bTPLX\b", "triplex"),
        (r"\bQUAD\b", "quadplex"),
        (r"\bWIC\b", "walk-in closet"),
        (r"\bFPLC\b", "fireplace"),
        (r"\bFP\b", "fireplace"),
        (r"\bW\s*/\s*D\b", "washer dryer"),
        (r"\bA\s*/\s*C\b", "air conditioning"),
        (r"\bMLS\b", "M L S"),
        (r"\bHOA\b", "H O A"),
        (r"\bADU\b", "A D U"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _normalize_real_estate_shorthand_for_speech(text: str) -> str:
    bed_bath_re = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:bd|br|beds?|bedrooms?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:ba|baths?|bathrooms?)\b",
        flags=re.IGNORECASE,
    )

    def _bed_bath_sub(m: re.Match[str]) -> str:
        beds = _bedroom_phrase(m.group(1))
        baths = _bathroom_phrase(m.group(2))
        return f"{beds}, {baths}"

    text = bed_bath_re.sub(_bed_bath_sub, text)
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*(?:bd|br)\b",
        lambda m: _bedroom_phrase(m.group(1)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*ba\b",
        lambda m: _bathroom_phrase(m.group(1)),
        text,
        flags=re.IGNORECASE,
    )
    return text


def _normalize_measurements_for_speech(text: str) -> str:
    text = re.sub(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|sqft)\b",
        lambda m: f"{_number_token_to_words(m.group(1))} square feet",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:acres?)\b",
        lambda m: _measurement_phrase(m.group(1), "acre"),
        text,
        flags=re.IGNORECASE,
    )
    return text


_ADDRESS_SUFFIX_MAP = {
    "aly": "Alley",
    "ave": "Avenue",
    "blvd": "Boulevard",
    "bnd": "Bend",
    "cir": "Circle",
    "ct": "Court",
    "cv": "Cove",
    "dr": "Drive",
    "expy": "Expressway",
    "fwy": "Freeway",
    "gln": "Glen",
    "grn": "Green",
    "hwy": "Highway",
    "ln": "Lane",
    "pkwy": "Parkway",
    "pl": "Place",
    "plz": "Plaza",
    "pt": "Point",
    "rd": "Road",
    "rdg": "Ridge",
    "sq": "Square",
    "st": "Street",
    "ter": "Terrace",
    "trl": "Trail",
    "vw": "View",
    "xing": "Crossing",
}

_ADDRESS_SUFFIX_PATTERN = "|".join(
    sorted((re.escape(key) for key in _ADDRESS_SUFFIX_MAP), key=len, reverse=True)
)
_ADDRESS_FOLLOWUP_TOKENS = (
    r"#|Apt\.?|Apartment|Unit|Suite|Ste\.?|Room|Rm\.?|Floor|Fl\.?|"
    r"and|in|near|at|on|off|for|with|by|to|from|newly|nearby|listed|listing|"
    r"active|pending|city|state|CA|\d{5}|"
    r"[a-z][a-z0-9-]*"
)
_ADDRESS_SUFFIX_RE = re.compile(
    rf"(?P<prefix>\b(?:\d+[A-Za-z]?\s+)?(?:[A-Z0-9][A-Za-z0-9'’.-]*\s+){{1,4}})"
    rf"(?P<suffix>{_ADDRESS_SUFFIX_PATTERN})\.?\b"
    rf"(?=(?:\s*(?:[,.;:)]|$))|\s+(?:{_ADDRESS_FOLLOWUP_TOKENS})\b)",
    flags=re.IGNORECASE,
)

_STATE_ABBREVIATION_MAP = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
_STATE_ABBREVIATION_PATTERN = "|".join(
    sorted((re.escape(key) for key in _STATE_ABBREVIATION_MAP), key=len, reverse=True)
)
_STATE_ABBREVIATION_RE = re.compile(
    rf"(?P<prefix>,\s*)(?P<state>{_STATE_ABBREVIATION_PATTERN})\b"
    rf"(?P<zip>\s+\d{{5}}(?:-\d{{4}})?)?"
    rf"(?=(?:\s*(?:[,.;:)\]]|[-–—]|$)))",
    flags=re.IGNORECASE,
)


def _normalize_address_suffixes_for_speech(text: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        suffix = match.group("suffix").rstrip(".").lower()
        replacement = _ADDRESS_SUFFIX_MAP.get(suffix)
        if not replacement:
            return match.group(0)
        last_token = prefix.strip().split()[-1]
        is_street_like = any(ch.isdigit() for ch in prefix) or any(
            ch.isupper() for ch in last_token
        )
        if not is_street_like:
            return match.group(0)
        return f"{prefix}{replacement}"

    return _ADDRESS_SUFFIX_RE.sub(_sub, text)


def _normalize_state_abbreviations_for_speech(text: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        state = match.group("state").upper()
        replacement = _STATE_ABBREVIATION_MAP.get(state)
        if not replacement:
            return match.group(0)
        return f"{match.group('prefix')}{replacement}{match.group('zip') or ''}"

    return _STATE_ABBREVIATION_RE.sub(_sub, text)


def _measurement_phrase(number_token: str, singular: str) -> str:
    normalized = number_token.replace(",", "")
    try:
        quantity = float(normalized)
    except ValueError:
        return f"{number_token} {singular}s"
    label = singular if quantity == 1 else f"{singular}s"
    return f"{_number_token_to_words(number_token)} {label}"


def _bedroom_phrase(number_token: str) -> str:
    return f"{_number_token_to_words(number_token)} bedroom"


def _bathroom_phrase(number_token: str) -> str:
    raw = number_token.replace(",", "").strip()
    if "." not in raw:
        return f"{_number_token_to_words(raw)} bath"

    whole, frac = raw.split(".", 1)
    frac = frac.rstrip("0")
    special_fractions = {
        "25": "and a quarter",
        "5": "and a half",
        "50": "and a half",
        "75": "and three quarter",
    }
    if frac in special_fractions:
        whole_words = _num_to_words_int(int(whole or "0"))
        return f"{whole_words} {special_fractions[frac]} bath"

    return f"{_number_token_to_words(raw)} bath"


class StreamSanitizer:
    """
    Stateful sanitizer for streamed assistant content.

    Two modes match the upstream OpenClaw orchestrator's published contract
    (system-prompt-BIKbdIsV.js): "Format every reply as <think>...</think>
    then <final>...</final>, with no other text. Only text inside <final> is
    shown to the user; everything else is discarded and never seen by the
    user."

    - strict_final=True (use for openclaw:* gateway models): emit ONLY the
      content between <final> and </final>. Drop everything else, including
      reasoning prose that arrived without <think> wrapping (Gemini native
      thinking that leaks through the gateway's OpenAI-compat stream).

    - strict_final=False (use for direct OpenAI / non-orchestrator backends):
      emit everything except content inside <think>...</think>.

    Both modes are robust to tag fragments split across chunk boundaries —
    e.g. "<thi" + "nk>", "<fina" + "l>", "</fina" + "l>".
    """

    __slots__ = ("strict_final", "_buf", "_state")

    _OUTSIDE = "outside"
    _IN_THINK = "in_think"
    _IN_FINAL = "in_final"

    def __init__(self, strict_final: bool):
        self.strict_final = strict_final
        self._buf = ""
        self._state = self._OUTSIDE

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if self._state == self._IN_THINK:
                close = _THINK_CLOSE_RE.search(self._buf)
                if not close:
                    if len(self._buf) > len("</think>"):
                        self._buf = self._buf[-(len("</think>") - 1) :]
                    break
                self._buf = self._buf[close.end() :]
                self._state = self._OUTSIDE
                continue

            if self._state == self._IN_FINAL:
                close = _FINAL_CLOSE_RE.search(self._buf)
                if not close:
                    tail_keep = min(len(self._buf), len("</final>") - 1)
                    emit_to = len(self._buf) - tail_keep
                    if emit_to > 0:
                        out.append(self._buf[:emit_to])
                        self._buf = self._buf[emit_to:]
                    break
                out.append(self._buf[: close.start()])
                self._buf = self._buf[close.end() :]
                self._state = self._OUTSIDE
                continue

            consumed = self._consume_outside(out)
            if not consumed:
                break

        return "".join(out)

    def _consume_outside(self, out: list[str]) -> bool:
        """
        Process buffer in the OUTSIDE state. Returns True if it made forward
        progress (loop should continue), False if it needs more input.
        """
        think_match = _THINK_OPEN_RE.search(self._buf)
        final_match = _FINAL_OPEN_RE.search(self._buf)

        candidates = []
        if think_match:
            candidates.append((think_match.start(), think_match.end(), self._IN_THINK))
        if final_match:
            candidates.append((final_match.start(), final_match.end(), self._IN_FINAL))

        if candidates:
            start, end, next_state = min(candidates, key=lambda c: c[0])
            if start > 0 and not self.strict_final:
                out.append(self._buf[:start])
            self._buf = self._buf[end:]
            self._state = next_state
            return True

        lt_idx = self._buf.find("<")

        if lt_idx == -1:
            if not self.strict_final and self._buf:
                out.append(self._buf)
            self._buf = ""
            return False

        if lt_idx > 0:
            prefix = self._buf[:lt_idx]
            if not self.strict_final:
                out.append(prefix)
            self._buf = self._buf[lt_idx:]

        suffix_lower = self._buf.lower()
        if (
            "<think".startswith(suffix_lower)
            or "<final".startswith(suffix_lower)
            or suffix_lower.startswith("<think")
            or suffix_lower.startswith("<final")
        ):
            return False

        if not self.strict_final:
            out.append(self._buf[0])
        self._buf = self._buf[1:]
        return True

    def flush(self) -> str:
        """
        Drain any remaining content. Call once after the stream has ended.

        - In strict mode: anything left in the OUTSIDE state is dropped
          (it was reasoning leakage). If we ended IN_FINAL without seeing
          </final>, emit the buffered content (the close tag was eaten by an
          upstream truncation).
        - In lenient mode: emit any pending text that wasn't a partial tag.
        """
        out = ""
        if self._state == self._IN_FINAL:
            out = self._buf
        elif self._state == self._OUTSIDE and not self.strict_final:
            tail = self._buf
            if tail and ("<think".startswith(tail.lower()) or "<final".startswith(tail.lower())):
                tail = ""
            out = tail
        self._buf = ""
        self._state = self._OUTSIDE
        return out


def estimate_speech_duration(text: str, wpm: int = 150) -> float:
    """
    Estimate speech duration in seconds.

    Args:
        text: Text to speak
        wpm: Words per minute (default 150 for natural speech)

    Returns:
        Estimated duration in seconds
    """
    word_count = len(text.split())
    return (word_count / wpm) * 60


_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _num_to_words_0_59(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[ones]}"


def _num_to_words_int(n: int) -> str:
    if n < 0:
        return f"minus {_num_to_words_int(abs(n))}"
    if n < 60:
        return _num_to_words_0_59(n)
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return _TENS[tens]
        return f"{_TENS[tens]} {_ONES[ones]}"
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        prefix = f"{_ONES[hundreds]} hundred"
        return prefix if remainder == 0 else f"{prefix} {_num_to_words_int(remainder)}"
    if n < 1_000_000:
        thousands = n // 1000
        remainder = n % 1000
        prefix = f"{_num_to_words_int(thousands)} thousand"
        return prefix if remainder == 0 else f"{prefix} {_num_to_words_int(remainder)}"
    if n < 1_000_000_000:
        millions = n // 1_000_000
        remainder = n % 1_000_000
        prefix = f"{_num_to_words_int(millions)} million"
        return prefix if remainder == 0 else f"{prefix} {_num_to_words_int(remainder)}"
    billions = n // 1_000_000_000
    remainder = n % 1_000_000_000
    prefix = f"{_num_to_words_int(billions)} billion"
    return prefix if remainder == 0 else f"{prefix} {_num_to_words_int(remainder)}"


def _number_token_to_words(token: str) -> str:
    raw = token.replace(",", "").strip()
    if not raw:
        return token

    if "." in raw:
        whole, frac = raw.split(".", 1)
        frac = frac.rstrip("0")
        if not frac:
            return _num_to_words_int(int(whole or "0"))
        whole_words = _num_to_words_int(int(whole or "0"))
        frac_words = " ".join(_ONES[int(ch)] for ch in frac if ch.isdigit())
        return f"{whole_words} point {frac_words}".strip()

    return _num_to_words_int(int(raw))


def _parse_time_token(token: str) -> Optional[tuple[int, int]]:
    """
    Parse a time token to (hour_24, minute).
    Supports:
    - 11:30
    - 11:30am / 11:30 pm
    - 2pm / 2 pm
    - 1130 / 200 / 14 (contextual use)
    """
    t = token.strip().lower()
    mer = None
    m_mer = re.search(r"([ap])\.?\s*m\.?$", t)
    if m_mer:
        mer = m_mer.group(1)
        t = re.sub(r"([ap])\.?\s*m\.?$", "", t).strip()

    hour: Optional[int] = None
    minute: Optional[int] = None

    if ":" in t:
        parts = t.split(":", 1)
        if not (parts[0].isdigit() and parts[1].isdigit()):
            return None
        hour = int(parts[0])
        minute = int(parts[1])
    elif t.isdigit():
        if len(t) == 4:
            hour = int(t[:2])
            minute = int(t[2:])
        elif len(t) == 3:
            hour = int(t[0])
            minute = int(t[1:])
        elif len(t) <= 2:
            hour = int(t)
            minute = 0
        else:
            return None
    else:
        return None

    if minute is None or hour is None:
        return None
    if minute < 0 or minute > 59:
        return None

    if mer:
        if hour < 1 or hour > 12:
            return None
        if mer == "a":
            hour24 = 0 if hour == 12 else hour
        else:
            hour24 = 12 if hour == 12 else hour + 12
    else:
        if hour < 0 or hour > 23:
            return None
        hour24 = hour

    return hour24, minute


def _time_to_words(hour24: int, minute: int) -> str:
    if hour24 == 0 and minute == 0:
        return "midnight"
    if hour24 == 12 and minute == 0:
        return "noon"

    hour12 = hour24 % 12
    hour_word = _ONES[12 if hour12 == 0 else hour12]

    if minute == 0:
        return hour_word
    if minute < 10:
        return f"{hour_word} oh {_ONES[minute]}"
    return f"{hour_word} {_num_to_words_0_59(minute)}"


_TIME_TOKEN_PATTERN = r"(?:[01]?\d|2[0-3])(?::[0-5]\d)?(?:\s*[AaPp]\.?\s*[Mm]\.?)?|(?:\d{3,4})(?:\s*[AaPp]\.?\s*[Mm]\.?)?"


def _normalize_times_for_speech(text: str) -> str:
    # 1) Time ranges first: "11:30-12:00", "1130 to 1200", "2pm–3pm"
    range_re = re.compile(
        rf"\b({_TIME_TOKEN_PATTERN})\b\s*(?:-|–|—|to)\s*\b({_TIME_TOKEN_PATTERN})\b",
        flags=re.IGNORECASE,
    )

    def _range_sub(m: re.Match[str]) -> str:
        left = _parse_time_token(m.group(1))
        right = _parse_time_token(m.group(2))
        if not left or not right:
            return m.group(0)
        return f"{_time_to_words(*left)} to {_time_to_words(*right)}"

    text = range_re.sub(_range_sub, text)

    # 2) Standalone explicit time tokens with ":" or am/pm.
    explicit_re = re.compile(
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*[AaPp]\.?\s*[Mm]\.?)?\b|\b(?:1[0-2]|0?[1-9])\s*[AaPp]\.?\s*[Mm]\.?\b",
        flags=re.IGNORECASE,
    )

    def _explicit_sub(m: re.Match[str]) -> str:
        parsed = _parse_time_token(m.group(0))
        if not parsed:
            return m.group(0)
        return _time_to_words(*parsed)

    text = explicit_re.sub(_explicit_sub, text)

    # 3) Contextual compact times like "at 200" / "from 1130"
    contextual_re = re.compile(r"\b(at|from|by|around)\s+(\d{3,4})\b", flags=re.IGNORECASE)

    def _context_sub(m: re.Match[str]) -> str:
        parsed = _parse_time_token(m.group(2))
        if not parsed:
            return m.group(0)
        return f"{m.group(1)} {_time_to_words(*parsed)}"

    return contextual_re.sub(_context_sub, text)
