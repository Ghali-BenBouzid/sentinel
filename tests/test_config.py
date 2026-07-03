"""Tests for the pydantic-settings config layer.

Covers the three things that are ours to get right: the free-tier default,
env-over-.env precedence, and that the GROK_API_KEY alias populates the same
Groq key as GROQ_API_KEY. `get_settings` is cached, so each test builds a fresh
`Settings` directly (or clears the cache) to stay isolated.
"""

from __future__ import annotations

import pytest

from sentinel.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Keep the cached settings from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_free_tier(monkeypatch, tmp_path):
    # No env, no .env file -> free-tier default, no keys.
    for var in ("SENTINEL_LLM_PROVIDER", "GROQ_API_KEY", "GROK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # ignore any real .env on disk
    assert settings.sentinel_llm_provider == "groq"
    assert settings.groq_api_key is None
    assert settings.anthropic_api_key is None


def test_grok_alias_populates_groq_key(monkeypatch):
    # The captain's habitual GROK spelling must fill the same Groq key.
    for var in ("GROQ_API_KEY", "GROK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROK_API_KEY", "grok-typo-key")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "grok-typo-key"


def test_canonical_name_still_works(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "canonical-key")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "canonical-key"


def test_env_overrides_dotenv(monkeypatch, tmp_path):
    # .env sets one value; a real env var must win over it.
    env_file = tmp_path / ".env"
    env_file.write_text("SENTINEL_LLM_PROVIDER=anthropic\nGROQ_API_KEY=from-dotenv\n")
    monkeypatch.delenv("SENTINEL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Only .env: its values are read.
    from_file = Settings(_env_file=str(env_file))
    assert from_file.sentinel_llm_provider == "anthropic"
    assert from_file.groq_api_key == "from-dotenv"

    # Real env var takes precedence over the same key in .env.
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    overridden = Settings(_env_file=str(env_file))
    assert overridden.sentinel_llm_provider == "groq"  # env wins
    assert overridden.groq_api_key == "from-dotenv"  # untouched .env value


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
