"""12-factor config for the agent layer.

Config comes from the environment and from a `.env` file (env wins over `.env`;
pydantic-settings handles the precedence). This is the single place the LLM
provider seam reads its provider choice and API keys from - the captain sets his
key in `.env` and runs M2 live, no `export` needed.

The Groq key deliberately accepts both `GROQ_API_KEY` (canonical) and
`GROK_API_KEY` (alias) - "Groq" (groq.com) is easy to mistype as xAI's "Grok".
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App config, read from the environment and a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Which LLM provider the seam uses. Default keeps the free-tier, zero-cost path.
    sentinel_llm_provider: str = "groq"

    # Free-tier provider key (groq.com). Accepts GROQ_API_KEY or the GROK_API_KEY
    # alias so a habitual "GROK" typo still populates the same key.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "GROK_API_KEY"),
    )

    # Paid Claude provider key (only needed when SENTINEL_LLM_PROVIDER=anthropic).
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )

    # SQLite file the LangGraph checkpointer persists resumable run state to.
    checkpoint_db_path: str = "artifacts/sentinel-checkpoints.sqlite"

    # V2 agent autonomy: guarded confirms expensive/destructive tools.
    sentinel_autonomy: str = "guarded"

    # Optional per-tier model-name overrides.
    sentinel_model_smart: str | None = None
    sentinel_model_cheap: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached `Settings` (env + `.env`)."""
    return Settings()
