"""Custom reliability middleware for the agent harness.

Design: docs/superpowers/specs/2026-07-11-agent-harness-middleware.md
"""
from __future__ import annotations

import re
from collections.abc import Callable

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from .registry import Registry

_MISSING_PROPERTIES = re.compile(r"missing properties:\s*(.+?)\]")
_QUOTED = re.compile(r"'([^']+)'")


class ModelFailureFormatterMiddleware(AgentMiddleware):
    """Turn exhausted model-call failures into a graceful assistant reply."""

    def __init__(self, corrective_feedback: Callable[[Exception], str]) -> None:
        super().__init__()
        self._corrective_feedback = corrective_feedback

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as error:  # noqa: BLE001
            return AIMessage(content=self._corrective_feedback(error))

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:  # noqa: BLE001
            return AIMessage(content=self._corrective_feedback(error))


def _tier1_template(error: Exception, registry: Registry) -> str | None:
    """Return deterministic feedback for a recognized error shape."""
    text = str(error)
    if (match := _MISSING_PROPERTIES.search(text)) is not None:
        fields = _QUOTED.findall(match.group(1))
        if fields:
            return (
                f"That call is missing required fields: {', '.join(fields)}. "
                "Ask the user for these before calling it again."
            )
    if isinstance(error, KeyError):
        (bad_id,) = error.args or ("?",)
        known = registry.list()
        return (
            f"'{bad_id}' is not a registered model id. "
            f"Known model ids: {known or '(none registered yet)'}."
        )
    return None


def _tier2_cheap_model(error: Exception, tools_chat_model: BaseChatModel) -> str:
    """Ask the cheap-tier model to summarize an unrecognized error."""
    prompt = (
        "An internal call just failed with this error:\n\n"
        f"{error}\n\n"
        "In one sentence, tell the agent what went wrong and what to try next. "
        "Do not mention exception types or stack traces."
    )
    response = tools_chat_model.invoke([HumanMessage(prompt)])
    return str(response.content)


def make_corrective_feedback(
    tools_chat_model: BaseChatModel, registry: Registry
) -> Callable[[Exception], str]:
    """Build the shared deterministic-first corrective-feedback function."""

    def corrective_feedback(error: Exception) -> str:
        templated = _tier1_template(error, registry)
        if templated is not None:
            return templated
        return _tier2_cheap_model(error, tools_chat_model)

    return corrective_feedback
