"""Offline test doubles for the V2 agent."""
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class FakeChatModel(GenericFakeChatModel):
    """A scripted chat model that ignores bound tools."""

    def bind_tools(self, *args, **kwargs):
        return self
