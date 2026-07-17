"""Bounded, request-only projection of durable agent message history."""
from __future__ import annotations

from copy import deepcopy

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage


class BoundedToolContextMiddleware(AgentMiddleware):
    """Clear stale tool payloads in model requests without mutating state."""

    def __init__(
        self,
        *,
        trigger_tokens: int,
        clear_at_least_tokens: int,
        keep_tool_results: int,
        placeholder: str,
    ) -> None:
        self.trigger_tokens = trigger_tokens
        self.clear_at_least_tokens = clear_at_least_tokens
        self.keep_tool_results = keep_tool_results
        self.placeholder = placeholder

    @staticmethod
    def _tokens(messages) -> int:
        # A deterministic approximation avoids provider tokenizers and network
        # access at the request seam. Four characters per token is conservative
        # for the English prose and JSON Sentinel transports.
        return sum(len(str(message.content)) for message in messages) // 4

    def _project(self, messages):
        projected = deepcopy(list(messages))
        initial_tokens = self._tokens(projected)
        if initial_tokens <= self.trigger_tokens:
            return projected

        candidates = [
            index
            for index, message in enumerate(projected)
            if isinstance(message, ToolMessage)
        ]
        if self.keep_tool_results:
            candidates = candidates[: -self.keep_tool_results]

        for index in candidates:
            message = projected[index]
            projected[index] = message.model_copy(
                update={
                    "artifact": None,
                    "content": self.placeholder,
                    "response_metadata": {
                        **message.response_metadata,
                        "context_projection": {"cleared": True},
                    },
                }
            )
            if (
                initial_tokens - self._tokens(projected)
                >= self.clear_at_least_tokens
            ):
                break
        return projected

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return handler(request.override(messages=self._project(request.messages)))
