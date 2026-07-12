"""Custom reliability middleware for the agent harness.

Design: docs/superpowers/specs/2026-07-11-agent-harness-middleware.md
"""
from __future__ import annotations

import re
from collections.abc import Callable

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .registry import Registry

_MISSING_PROPERTIES = re.compile(r"missing properties:\s*(.+?)\]")
_QUOTED = re.compile(r"'([^']+)'")
_AUTH_ERROR = re.compile(
    r"api[_ -]?key|auth[_ -]?token|authentication failed|invalid token|"
    r"request headers",
    re.IGNORECASE,
)


def guarded_when(tool_name: str):
    """Interrupt guarded sessions and audit auto-approved autonomous calls."""

    def when(request) -> bool:
        if request.state.get("autonomy") == "autonomous":
            request.runtime.stream_writer({
                "type": "auto_approved",
                "tool": tool_name,
                "detail": str(request.tool_call["args"]),
            })
            return False
        return True

    return when


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


class InvalidToolCallMiddleware(AgentMiddleware):
    """Return malformed parsed tool calls to the model with corrective feedback."""

    def __init__(self, corrective_feedback: Callable[[Exception], str]) -> None:
        super().__init__()
        self._corrective_feedback = corrective_feedback

    def after_model(self, state, runtime):
        messages = state["messages"]
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.invalid_tool_calls:
            return None

        tool_messages = []
        for invalid in last.invalid_tool_calls:
            tool_messages.append(ToolMessage(
                content=self._corrective_feedback(
                    ValueError(invalid.get("error") or "the tool call could not be parsed")
                ),
                tool_call_id=invalid.get("id") or "unknown",
                name=invalid.get("name") or "unknown_tool",
                status="error",
            ))
        return {"messages": tool_messages, "jump_to": "model"}

def _tier1_template(error: Exception, registry: Registry) -> str | None:
    """Return deterministic feedback for a recognized error shape."""
    text = str(error)
    if _AUTH_ERROR.search(text):
        return (
            "The language-model provider configuration is unavailable or "
            "invalid. Retry after the operator verifies provider configuration."
        )
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


_CORRECTIVE_SYSTEM_PROMPT = (
    "You produce private corrective feedback for an autonomous agent after an "
    "internal model or tool call fails. Identify only what the supplied error "
    "supports. Give one concrete action the agent itself can take next. Never "
    "invent missing values, claim the failed action succeeded, expose a stack "
    "trace, or instruct the end user to invoke internal tools. Return exactly "
    "one concise sentence with no preamble."
)


def _tier2_cheap_model(error: Exception, tools_chat_model: BaseChatModel) -> str:
    """Ask the cheap-tier model to summarize an unrecognized error."""
    prompt = (
        "An internal call just failed with this error:\n\n"
        f"{error}\n\n"
        "State what failed and the safest supported next action."
    )
    response = tools_chat_model.invoke(
        [SystemMessage(_CORRECTIVE_SYSTEM_PROMPT), HumanMessage(prompt)]
    )
    return str(response.content)


def make_corrective_feedback(
    tools_chat_model: BaseChatModel, registry: Registry
) -> Callable[[Exception], str]:
    """Build the shared deterministic-first corrective-feedback function."""

    def corrective_feedback(error: Exception) -> str:
        templated = _tier1_template(error, registry)
        if templated is not None:
            return templated
        try:
            return _tier2_cheap_model(error, tools_chat_model)
        except Exception:  # noqa: BLE001
            return (
                "The language-model service is temporarily unavailable. "
                "Retry shortly; if it persists, verify provider configuration."
            )

    return corrective_feedback
