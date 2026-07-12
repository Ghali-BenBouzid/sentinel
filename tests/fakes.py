"""Offline test doubles for the V2 agent."""
from __future__ import annotations

import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class FakeChatModel(GenericFakeChatModel):
    """A scripted chat model that ignores bound tools."""

    def bind_tools(self, *args, **kwargs):
        return self


class SlowFakeChatModel(FakeChatModel):
    """A scripted chat model whose sync call blocks, like a real HTTP round trip.

    Used to prove requests are handled concurrently rather than serialized
    behind one blocking generate() call.
    """

    delay_seconds: float = 1.0

    def _generate(self, *args, **kwargs):
        time.sleep(self.delay_seconds)
        return super()._generate(*args, **kwargs)


class RaisingThenFakeChatModel(FakeChatModel):
    """Raise a scripted exception first, then behave normally."""

    fail_times: int = 1
    exception_factory: object = None

    def _generate(self, *args, **kwargs):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.exception_factory()
        return super()._generate(*args, **kwargs)
