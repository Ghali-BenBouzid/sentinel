"""12-factor config for the agent layer.

Config comes from the environment and from a `.env` file (env wins over `.env`;
pydantic-settings handles the precedence). This is the single place the LLM
provider seam reads its provider choice and API keys from - the captain sets his
key in `.env` and runs M2 live, no `export` needed.

The Groq key deliberately accepts both `GROQ_API_KEY` (canonical) and
`GROK_API_KEY` (alias) - "Groq" (groq.com) is easy to mistype as xAI's "Grok".
"""

from __future__ import annotations

import os
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

    # Harness middleware: runaway-loop / cost insurance.
    sentinel_model_call_thread_limit: int = 40
    sentinel_model_call_run_limit: int = 15
    sentinel_tool_call_thread_limit: int = 40
    sentinel_tool_call_run_limit: int = 20

    # Harness middleware: retry attempts for a failed model call.
    sentinel_retry_max_attempts: int = 2

    # LangSmith reads these canonical names directly from os.environ. Sentinel
    # keeps them in its settings layer first, then exports only this allowlist
    # at process entrypoints via configure_langsmith().
    sentinel_langsmith_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SENTINEL_LANGSMITH_API_KEY", "LANGSMITH_API_KEY"
        ),
    )
    sentinel_langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "SENTINEL_LANGSMITH_TRACING", "LANGSMITH_TRACING"
        ),
    )
    sentinel_langsmith_project: str = Field(
        default="sentinel",
        validation_alias=AliasChoices(
            "SENTINEL_LANGSMITH_PROJECT", "LANGSMITH_PROJECT"
        ),
    )
    sentinel_langsmith_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SENTINEL_LANGSMITH_ENDPOINT", "LANGSMITH_ENDPOINT"
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached `Settings` (env + `.env`)."""
    return Settings()


def configure_langsmith(settings: Settings | None = None) -> None:
    """Export Sentinel's LangSmith allowlist for the LangSmith SDK."""
    settings = settings or get_settings()
    values = {
        "LANGSMITH_API_KEY": settings.sentinel_langsmith_api_key,
        "LANGSMITH_TRACING": str(settings.sentinel_langsmith_tracing).lower(),
        "LANGSMITH_PROJECT": settings.sentinel_langsmith_project,
        "LANGSMITH_ENDPOINT": settings.sentinel_langsmith_endpoint,
    }
    for name, value in values.items():
        if value is not None:
            os.environ[name] = value
