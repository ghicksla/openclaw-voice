from src.server.session_registry import ResumeRecord, SessionResumeRegistry


def test_resume_returns_matching_unexpired_session():
    registry = SessionResumeRegistry(grace_seconds=10)
    registry.park(ResumeRecord("client", "session", mode="continuous"), now=2)
    record = registry.resume("client", "session", now=5)
    assert record is not None
    assert record.mode == "continuous"


def test_resume_rejects_expired_or_mismatched_session():
    registry = SessionResumeRegistry(grace_seconds=10)
    registry.park(ResumeRecord("client", "session"), now=2)
    assert registry.resume("client", "other", now=5) is None

    registry.park(ResumeRecord("client", "session"), now=2)
    assert registry.resume("client", "session", now=12) is None


def test_capacity_evicts_earliest_expiry():
    registry = SessionResumeRegistry(grace_seconds=10, capacity=2)
    registry.park(ResumeRecord("first", "one"), now=1)
    registry.park(ResumeRecord("second", "two"), now=2)
    registry.park(ResumeRecord("third", "three"), now=3)
    assert registry.resume("first", "one", now=4) is None
    assert registry.resume("second", "two", now=4) is not None


def test_zero_grace_disables_parking():
    registry = SessionResumeRegistry(grace_seconds=0)
    registry.park(ResumeRecord("client", "session"), now=1)
    assert registry.resume("client", "session", now=1) is None
