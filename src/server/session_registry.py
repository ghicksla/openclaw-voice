"""Short-lived browser resume registry.

This complements, but never replaces, the durable delivery/outbox state keyed
by OpenClaw session_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResumeRecord:
    client_id: str
    session_id: str
    mode: str = "push_to_talk"
    turn_config: dict = field(default_factory=dict)
    expires_at: float = 0.0


class SessionResumeRegistry:
    def __init__(self, grace_seconds: int = 600, capacity: int = 32):
        self.grace_seconds = min(max(int(grace_seconds), 0), 3600)
        self.capacity = max(1, int(capacity))
        self._records: dict[str, ResumeRecord] = {}

    def resume(
        self, client_id: str, session_id: str, now: Optional[float] = None
    ) -> Optional[ResumeRecord]:
        now = time.monotonic() if now is None else now
        self.purge(now)
        record = self._records.pop(client_id, None)
        if record is None or record.session_id != session_id:
            return None
        return record

    def park(self, record: ResumeRecord, now: Optional[float] = None) -> None:
        if not record.client_id or self.grace_seconds <= 0:
            return
        now = time.monotonic() if now is None else now
        self.purge(now)
        record.expires_at = now + self.grace_seconds
        self._records[record.client_id] = record
        if len(self._records) > self.capacity:
            oldest = min(self._records.values(), key=lambda item: item.expires_at)
            self._records.pop(oldest.client_id, None)

    def remove(self, client_id: str) -> None:
        self._records.pop(client_id, None)

    def purge(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        expired = [
            client_id for client_id, record in self._records.items() if record.expires_at <= now
        ]
        for client_id in expired:
            self._records.pop(client_id, None)
