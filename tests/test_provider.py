"""The LLM seam returns a configured LangChain chat model."""
from __future__ import annotations

import pytest


def _clear():
    from sentinel.config import get_settings

    get_settings.cache_clear()


def test_get_chat_model_groq_default(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _clear()
    from langchain_groq import ChatGroq

    from sentinel.llm.provider import get_chat_model

    model = get_chat_model("smart")
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"


def test_groq_models_are_not_the_deprecated_ones():
    from sentinel.llm.provider import _MODELS

    assert _MODELS["groq"]["smart"] == "openai/gpt-oss-120b"
    assert _MODELS["groq"]["cheap"] == "openai/gpt-oss-20b"


def test_get_chat_model_anthropic(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _clear()
    from langchain_anthropic import ChatAnthropic

    from sentinel.llm.provider import get_chat_model

    assert isinstance(get_chat_model("cheap"), ChatAnthropic)


def test_model_override_from_settings(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SENTINEL_MODEL_SMART", "llama-3.1-8b-instant")
    _clear()
    from sentinel.llm.provider import get_chat_model

    assert get_chat_model("smart").model_name == "llama-3.1-8b-instant"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "bogus")
    _clear()
    from sentinel.llm.provider import get_chat_model

    with pytest.raises(ValueError):
        get_chat_model("smart")
