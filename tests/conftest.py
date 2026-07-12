"""Suite-wide isolation from developer-local observability configuration."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_langsmith_tracing(monkeypatch):
    """Offline tests must never upload traces using a developer's `.env`."""
    from sentinel.config import get_settings

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
