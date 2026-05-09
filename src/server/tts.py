"""
Text-to-Speech backends.

Priority:
1) ElevenLabs (when ELEVENLABS_API_KEY is present) - preferred
2) OpenAI TTS (when OPENAI_API_KEY is present)
3) Mock (no cloud key available)
"""

import asyncio
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from xml.sax.saxutils import escape as xml_escape

from loguru import logger


class ChatterboxTTS:
    """TTS adapter used by main.py (name kept for compatibility)."""

    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,
    ):
        # ElevenLabs settings (default backend)
        self._eleven_key: Optional[str] = None
        self._eleven_voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
        self._eleven_model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
        self._eleven_output_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
        self._eleven_pronunciation_dictionary_name = os.environ.get(
            "ELEVENLABS_PRONUNCIATION_DICTIONARY_NAME",
            "openclaw-voice",
        )
        self._eleven_pronunciation_rules_path = Path(
            os.environ.get(
                "ELEVENLABS_PRONUNCIATION_RULES_PATH",
                str(Path(__file__).resolve().with_name("pronunciation-rules.json")),
            )
        )
        self._eleven_pronunciation_dictionary_locators: list[dict[str, str]] = []

        # OpenAI fallback settings
        self._openai_key: Optional[str] = None
        self._openai_model = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
        self._openai_voice = os.environ.get("TTS_VOICE", "nova")
        self._openai_speed = float(os.environ.get("TTS_SPEED", "1.1"))

        self._backend = "mock"
        self._mime_type = "audio/aac"
        self._load_model()

    @property
    def mime_type(self) -> str:
        return self._mime_type

    def _load_model(self):
        # Prefer ElevenLabs if configured in service env.
        eleven_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if eleven_key:
            self._eleven_key = eleven_key
            self._backend = "elevenlabs"
            self._mime_type = "audio/mpeg" if self._eleven_output_format.startswith("mp3") else "audio/wav"
            self._sync_elevenlabs_pronunciation_dictionary()
            logger.info(
                "✅ ElevenLabs TTS ready "
                f"(voice={self._eleven_voice_id}, model={self._eleven_model}, format={self._eleven_output_format})"
            )
            return

        # Fallback to OpenAI when present.
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            self._openai_key = openai_key
            self._backend = "openai"
            self._mime_type = "audio/aac"
            logger.info(
                f"✅ OpenAI TTS ready (model={self._openai_model}, voice={self._openai_voice}, speed={self._openai_speed})"
            )
            return

        logger.warning("⚠️ No ELEVENLABS_API_KEY or OPENAI_API_KEY found — TTS disabled (mock mode)")
        self._backend = "mock"

    def _load_pronunciation_rules(self) -> list[dict[str, Any]]:
        if not self._eleven_pronunciation_rules_path.exists():
            return []

        try:
            raw = json.loads(self._eleven_pronunciation_rules_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load pronunciation rules: {e}")
            return []

        if not isinstance(raw, list):
            logger.error("Pronunciation rules file must contain a JSON array")
            return []

        rules: list[dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(f"Skipping pronunciation rule #{idx + 1}: expected object")
                continue

            string_to_replace = str(item.get("string_to_replace", "")).strip()
            rule_type = str(item.get("type", "alias")).strip().lower()
            if not string_to_replace:
                logger.warning(f"Skipping pronunciation rule #{idx + 1}: missing string_to_replace")
                continue

            rule: dict[str, Any] = {
                "string_to_replace": string_to_replace,
                "type": rule_type,
                "case_sensitive": bool(item.get("case_sensitive", False)),
                "word_boundaries": bool(item.get("word_boundaries", True)),
            }
            if rule_type == "alias":
                alias = str(item.get("alias", "")).strip()
                if not alias:
                    logger.warning(f"Skipping pronunciation alias rule #{idx + 1}: missing alias")
                    continue
                rule["alias"] = alias
            elif rule_type == "phoneme":
                phoneme = str(item.get("phoneme", "")).strip()
                alphabet = str(item.get("alphabet", "")).strip().lower()
                if not phoneme or not alphabet:
                    logger.warning(f"Skipping pronunciation phoneme rule #{idx + 1}: missing phoneme/alphabet")
                    continue
                rule["phoneme"] = phoneme
                rule["alphabet"] = alphabet
            else:
                logger.warning(f"Skipping pronunciation rule #{idx + 1}: unsupported type {rule_type!r}")
                continue

            rules.append(rule)

        return rules

    def _rules_to_pls(self, rules: list[dict[str, Any]]) -> bytes:
        lexemes: list[str] = []
        for rule in rules:
            grapheme = xml_escape(rule["string_to_replace"])
            if rule["type"] == "alias":
                alias = xml_escape(str(rule["alias"]))
                lexemes.append(
                    "  <lexeme>\n"
                    f"    <grapheme>{grapheme}</grapheme>\n"
                    f"    <alias>{alias}</alias>\n"
                    "  </lexeme>"
                )
            elif rule["type"] == "phoneme" and rule.get("alphabet") == "ipa":
                phoneme = xml_escape(str(rule["phoneme"]))
                lexemes.append(
                    "  <lexeme>\n"
                    f"    <grapheme>{grapheme}</grapheme>\n"
                    f"    <phoneme>{phoneme}</phoneme>\n"
                    "  </lexeme>"
                )

        contents = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lexicon version="1.0"\n'
            '    xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"\n'
            '    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '    xsi:schemaLocation="http://www.w3.org/2005/01/pronunciation-lexicon '
            'http://www.w3.org/TR/2007/CR-pronunciation-lexicon-20071212/pls.xsd"\n'
            '    alphabet="ipa" xml:lang="en-US">\n'
            f"{chr(10).join(lexemes)}\n"
            "</lexicon>\n"
        )
        return contents.encode("utf-8")

    def _rules_signature(self, rules: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        signature: list[tuple[str, str, str]] = []
        for rule in rules:
            grapheme = str(rule.get("string_to_replace", "")).strip()
            if rule.get("type") == "alias":
                signature.append((grapheme, "alias", str(rule.get("alias", "")).strip()))
            elif rule.get("type") == "phoneme":
                alphabet = str(rule.get("alphabet", "")).strip().lower()
                signature.append((grapheme, f"phoneme:{alphabet}", str(rule.get("phoneme", "")).strip()))
        return sorted(signature)

    def _pls_signature(self, pls_bytes: bytes) -> list[tuple[str, str, str]]:
        def _local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        try:
            root = ET.fromstring(pls_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to parse pronunciation dictionary PLS: {e}") from e

        signature: list[tuple[str, str, str]] = []
        for lexeme in root.iter():
            if _local_name(lexeme.tag) != "lexeme":
                continue

            grapheme = ""
            alias = ""
            phoneme = ""
            for child in lexeme:
                name = _local_name(child.tag)
                value = (child.text or "").strip()
                if name == "grapheme":
                    grapheme = value
                elif name == "alias":
                    alias = value
                elif name == "phoneme":
                    phoneme = value

            if grapheme and alias:
                signature.append((grapheme, "alias", alias))
            elif grapheme and phoneme:
                signature.append((grapheme, "phoneme:ipa", phoneme))

        return sorted(signature)

    def _sync_elevenlabs_pronunciation_dictionary(self):
        rules = self._load_pronunciation_rules()
        if not rules or not self._eleven_key:
            return

        dict_id: Optional[str] = None
        version_id: Optional[str] = None
        local_signature = self._rules_signature(rules)

        try:
            list_payload = self._elevenlabs_request_json(
                "GET",
                "/v1/pronunciation-dictionaries",
                params={"page_size": 100, "sort": "name", "sort_direction": "ascending"},
            )
            dictionaries = list_payload.get("pronunciation_dictionaries", [])
            existing = next(
                (item for item in dictionaries if item.get("name") == self._eleven_pronunciation_dictionary_name),
                None,
            )

            if existing:
                dict_id = existing.get("id")
                version_id = existing.get("latest_version_id")
                remote_signature = self._pls_signature(
                    self._elevenlabs_request_bytes(
                        "GET",
                        f"/v1/pronunciation-dictionaries/{dict_id}/{version_id}/download",
                    )
                )
                if remote_signature != local_signature:
                    update_payload = self._elevenlabs_request_json(
                        "POST",
                        f"/v1/pronunciation-dictionaries/{dict_id}/add-rules",
                        json_payload={"rules": rules},
                    )
                    version_id = update_payload.get("version_id") or version_id
            else:
                pls_bytes = self._rules_to_pls(rules)
                create_payload = self._elevenlabs_request_json(
                    "POST",
                    "/v1/pronunciation-dictionaries/add-from-file",
                    form_fields={
                        "name": self._eleven_pronunciation_dictionary_name,
                        "description": "OpenClaw Voice pronunciation aliases",
                    },
                    file_field=(
                        "file",
                        "openclaw-voice-pronunciation.pls",
                        pls_bytes,
                        "application/octet-stream",
                    ),
                )
                dict_id = create_payload.get("id")
                version_id = create_payload.get("version_id")

        except RuntimeError as e:
            logger.error(str(e))
            return
        except Exception as e:
            logger.error(f"Failed to sync ElevenLabs pronunciation dictionary: {e}")
            return

        if dict_id and version_id:
            self._eleven_pronunciation_dictionary_locators = [
                {
                    "pronunciation_dictionary_id": dict_id,
                    "version_id": version_id,
                }
            ]
            logger.info(
                "🔤 ElevenLabs pronunciation dictionary ready "
                f"(name={self._eleven_pronunciation_dictionary_name}, rules={len(rules)})"
            )

    async def synthesize_aac(self, text: str) -> Optional[bytes]:
        """
        Return encoded audio bytes for text.
        Name kept for compatibility with existing server code.
        """
        if self._backend == "mock":
            return None

        loop = asyncio.get_event_loop()
        if self._backend == "elevenlabs":
            return await loop.run_in_executor(None, self._synthesize_elevenlabs_sync, text)
        if self._backend == "openai":
            return await loop.run_in_executor(None, self._synthesize_openai_sync, text)
        return None

    def _synthesize_elevenlabs_sync(self, text: str) -> Optional[bytes]:
        payload: dict[str, Any] = {
            "text": text[:4000],
            "model_id": self._eleven_model,
            "voice_settings": {
                "stability": 0.35,
                "similarity_boost": 0.8,
            },
        }
        if self._eleven_pronunciation_dictionary_locators:
            payload["pronunciation_dictionary_locators"] = self._eleven_pronunciation_dictionary_locators

        try:
            return self._elevenlabs_request_bytes(
                "POST",
                f"/v1/text-to-speech/{self._eleven_voice_id}/stream",
                params={"output_format": self._eleven_output_format},
                json_payload=payload,
            )
        except RuntimeError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
        return None

    def _elevenlabs_request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_payload: Optional[dict[str, Any]] = None,
        form_fields: Optional[dict[str, str]] = None,
        file_field: Optional[tuple[str, str, bytes, str]] = None,
    ) -> dict[str, Any]:
        raw = self._elevenlabs_request_bytes(
            method,
            path,
            params=params,
            json_payload=json_payload,
            form_fields=form_fields,
            file_field=file_field,
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"ElevenLabs JSON decode error for {path}: {e}") from e

    def _elevenlabs_request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_payload: Optional[dict[str, Any]] = None,
        form_fields: Optional[dict[str, str]] = None,
        file_field: Optional[tuple[str, str, bytes, str]] = None,
    ) -> bytes:
        import uuid
        import urllib.error
        import urllib.request

        url = f"https://api.elevenlabs.io{path}"
        if params:
            url = f"{url}?{urlencode(params)}"

        headers = {
            "xi-api-key": self._eleven_key or "",
        }
        data: Optional[bytes] = None

        if json_payload is not None:
            data = json.dumps(json_payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif form_fields is not None or file_field is not None:
            boundary = f"----OpenClawBoundary{uuid.uuid4().hex}"
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            body = bytearray()

            for name, value in (form_fields or {}).items():
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                body.extend(str(value).encode("utf-8"))
                body.extend(b"\r\n")

            if file_field is not None:
                field_name, filename, contents, content_type = file_field
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(
                    (
                        f'Content-Disposition: form-data; name="{field_name}"; '
                        f'filename="{filename}"\r\n'
                    ).encode("utf-8")
                )
                body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
                body.extend(contents)
                body.extend(b"\r\n")

            body.extend(f"--{boundary}--\r\n".encode("utf-8"))
            data = bytes(body)

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"ElevenLabs HTTP {e.code} for {path}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"ElevenLabs request failed for {path}: {e}") from e

    def _synthesize_openai_sync(self, text: str) -> Optional[bytes]:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self._openai_model,
            "input": text[:4000],
            "voice": self._openai_voice,
            "speed": self._openai_speed,
            "response_format": "aac",
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._openai_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            logger.error(f"OpenAI TTS HTTP {e.code}: {e.read().decode()[:300]}")
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
        return None

    async def synthesize_stream(self, text: str):
        """Legacy interface kept for compatibility."""
        data = await self.synthesize_aac(text)
        if data:
            yield data
        else:
            yield b""
