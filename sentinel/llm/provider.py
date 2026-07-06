"""The LLM provider seam - now a LangChain chat model factory.

Tool-calling needs the richer ``bind_tools`` interface a plain
``complete() -> str`` seam cannot express, so this module returns a LangChain
chat model. Vendor-specific imports remain confined here.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import get_settings

_MODELS = {
    "anthropic": {"smart": "claude-sonnet-5", "cheap": "claude-haiku-4-5"},
    "groq": {
        "smart": "llama-3.3-70b-versatile",
        "cheap": "llama-3.1-8b-instant",
    },
}


def get_chat_model(
    tier: str = "smart", name: str | None = None
) -> BaseChatModel:
    """Build the configured chat model for a smart or cheap tier."""
    settings = get_settings()
    name = (name or settings.sentinel_llm_provider).lower()
    if name not in _MODELS:
        raise ValueError(
            f"unknown SENTINEL_LLM_PROVIDER {name!r}; "
            f"expected one of {list(_MODELS)}"
        )
    if tier not in ("smart", "cheap"):
        raise ValueError(
            f"unknown model tier {tier!r}; expected 'smart' or 'cheap'"
        )
    override = getattr(settings, f"sentinel_model_{tier}", None)
    model = override or _MODELS[name][tier]
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        max_tokens=1024,
    )
