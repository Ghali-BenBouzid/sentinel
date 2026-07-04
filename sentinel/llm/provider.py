"""The LLM provider seam.

This is deliberately *not* a routing or orchestration framework - it is one
interface with two implementations, so the agent graph can call an LLM without
importing a vendor SDK. Which provider is live is an env choice
(`SENTINEL_LLM_PROVIDER=anthropic|groq`), and each provider exposes two tiers:

- ``"cheap"`` - a small/fast model for the report writer (plain summarization).
- ``"smart"`` - a stronger model for the interviewer, where extracting structure
  from free text needs more judgment.

Both providers speak the same ``complete(messages, **kwargs) -> str`` shape.
``messages`` is a list of ``{"role": "system"|"user"|"assistant", "content": str}``
dicts (the OpenAI/Anthropic common shape); the string reply is returned as-is.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import get_settings

# Model tiers per provider. "smart" is used where judgment/extraction matters
# (interviewer); "cheap" for straightforward text generation (report writer).
_MODELS = {
    "anthropic": {"smart": "claude-sonnet-5", "cheap": "claude-haiku-4-5"},
    "groq": {"smart": "llama-3.3-70b-versatile", "cheap": "llama-3.1-8b-instant"},
}


@runtime_checkable
class Provider(Protocol):
    """Anything the graph can call to turn a chat into text."""

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Return the assistant's reply text for ``messages``."""
        ...


class AnthropicProvider:
    """Primary provider, backed by the `anthropic` SDK."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        # Import lazily so `pip install`ing only the free-tier SDK still works.
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, messages: list[dict], *, max_tokens: int = 1024, **kwargs) -> str:
        # Anthropic takes the system prompt as a separate arg, not a message.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or None,
            messages=chat,
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class GroqProvider:
    """Free-tier provider, backed by the OpenAI-compatible `groq` SDK.

    Chosen over the Gemini free tier for its lower-friction, OpenAI-shaped
    Python SDK. Needs a (free) `GROQ_API_KEY`.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key) if api_key else Groq()

    def complete(self, messages: list[dict], *, max_tokens: int = 1024, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


def get_provider(tier: str = "smart", name: str | None = None) -> Provider:
    """Build the configured provider for a model ``tier`` (``"smart"``/``"cheap"``).

    Provider choice and API keys come from `get_settings()` (env + `.env`); the
    resolved key is passed explicitly to the concrete provider. ``name`` overrides
    the configured provider. Raises ``ValueError`` for an unknown provider/tier -
    fail loud, don't guess.
    """
    settings = get_settings()
    name = (name or settings.sentinel_llm_provider).lower()
    if name not in _MODELS:
        raise ValueError(f"unknown SENTINEL_LLM_PROVIDER {name!r}; expected one of {list(_MODELS)}")
    if tier not in _MODELS[name]:
        raise ValueError(f"unknown model tier {tier!r}; expected 'smart' or 'cheap'")
    model = _MODELS[name][tier]
    if name == "anthropic":
        return AnthropicProvider(model, api_key=settings.anthropic_api_key)
    return GroqProvider(model, api_key=settings.groq_api_key)
