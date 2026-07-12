# Agent Harness Middleware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision note.** This plan was reviewed by Codex (`gpt-5.6-sol`, medium effort) against the real installed `langchain==1.3.11` source before implementation started, and revised from that review (see `docs/superpowers/specs/2026-07-11-agent-harness-middleware.md`'s revision note for the full account). The two changes that reshape task structure: `ToolRetryMiddleware` is dropped entirely (verified: `create_agent`'s `ToolNode` swallows tool exceptions into `ToolMessage`s before the middleware ever sees them, and `create_agent` exposes no way to turn that off), and the `HumanInTheLoopMiddleware` swap now includes a real frontend change (batched confirmation submission) plus CLI and snapshot-endpoint migrations that the first pass missed.

**Goal:** Close the reliability gap that let a malformed `save_config` tool call crash a turn and dump a raw exception to the user, and get Sentinel off two Groq models scheduled for shutdown - by giving every achievable failure category from `docs/learning/05-harness-engineering.html` a LangChain-native middleware.

**Architecture:** A new `sentinel/agents/harness.py` module holds the custom pieces (a shared `corrective_feedback` message-builder, `ModelFailureFormatterMiddleware`, `InvalidToolCallMiddleware`, and the `when` predicate that replaces `confirm()`'s autonomy check). `sentinel/agents/agent.py`'s `build_agent()` wires those alongside LangChain's built-in middleware (`ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ModelFallbackMiddleware`, `ModelRetryMiddleware`, `HumanInTheLoopMiddleware`) onto `create_agent(...)`. `sentinel/agents/tools.py` loses its hand-rolled `confirm()`/`interrupt()` calls. `sentinel/api/app.py`, `sentinel/agents/__main__.py`, and the frontend (`web/src/components/ConfirmCard.tsx`, `ChatView.tsx`, `web/src/state/useSession.ts`) all migrate from the old id-addressed, resolve-independently confirmation contract to `HumanInTheLoopMiddleware`'s bundled, all-at-once one.

**Tech Stack:** `langchain==1.3.11` (`langchain.agents.middleware`), `langgraph`, `groq==0.37.1`, existing Sentinel agent layer (FastAPI/SSE, pytest, offline fakes, React/TypeScript frontend, vitest).

## Global Constraints

- No live LLM, no PyCaret in the automated test suite - every test in this plan runs offline against fakes, per the project's existing testing discipline (`tests/fakes.py`).
- Config is 12-factor via `sentinel/config.py` (`get_settings()`, `lru_cache`d) - no hardcoded constants for anything a human might want to tune.
- Never import a vendor SDK (`groq`, `anthropic`) outside `sentinel/llm/provider.py`, except where a test needs to construct a real vendor exception type to make a fake realistic (`groq.BadRequestError`) - that's a test-only import, not app code.
- Tools return strings for expected errors, never raise, per the existing convention documented at the top of `sentinel/agents/tools.py`.
- Sentence-per-line formatting in any long Markdown this plan or its work touches, no em dashes, matching the user's global CLAUDE.md.
- Commit after every task's tests pass. Do not batch multiple tasks into one commit.

---

### Task 1: Swap the two deprecated Groq models

**Files:**
- Modify: `sentinel/llm/provider.py:13-19` (the `_MODELS` dict)
- Modify: `tests/test_provider.py:23` (stale assertion - Codex caught this: the file already asserts the old default and would fail after the swap)
- Test: `tests/test_provider.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_MODELS["groq"]["smart"] == "openai/gpt-oss-120b"`, `_MODELS["groq"]["cheap"] == "openai/gpt-oss-20b"` - later tasks' fallback/formatter wiring don't depend on the exact string, only that `get_chat_model` keeps working.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_provider.py`:

```python
def test_groq_models_are_not_the_deprecated_ones():
    from sentinel.llm.provider import _MODELS

    assert _MODELS["groq"]["smart"] == "openai/gpt-oss-120b"
    assert _MODELS["groq"]["cheap"] == "openai/gpt-oss-20b"
```

Also fix the existing `test_get_chat_model_groq_default` in the same file, whose assertion is currently pinned to the model this task retires:

```python
def test_get_chat_model_groq_default(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _clear()
    from langchain_groq import ChatGroq

    from sentinel.llm.provider import get_chat_model

    model = get_chat_model("smart")
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
```

`test_model_override_from_settings` (same file) sets `SENTINEL_MODEL_SMART=llama-3.1-8b-instant` explicitly via env var and asserts that exact override value comes back - it is testing the override mechanism, not the default, and needs no change.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider.py -v`
Expected: FAIL - `test_groq_models_are_not_the_deprecated_ones` fails (new), and `test_get_chat_model_groq_default` fails on its updated assertion against the still-old `_MODELS` dict.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/llm/provider.py`, change:

```python
_MODELS = {
    "anthropic": {"smart": "claude-sonnet-5", "cheap": "claude-haiku-4-5"},
    "groq": {
        "smart": "openai/gpt-oss-120b",
        "cheap": "openai/gpt-oss-20b",
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider.py -v`
Expected: PASS, all tests in the file green.

- [ ] **Step 5: Live smoke test (manual, not part of the automated suite)**

With a real `GROQ_API_KEY` in `.env`, run:

```bash
uv run python -c "
from sentinel.llm.provider import get_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def ping(x: int) -> str:
    '''Echo x.'''
    return str(x)

model = get_chat_model('smart').bind_tools([ping])
result = model.invoke([HumanMessage('Call ping with x=7.')])
print(result.tool_calls)
"
```

Expected: a non-empty `tool_calls` list naming `ping` with `{"x": 7}` (or similar) - confirms `openai/gpt-oss-120b` actually completes tool calls through `ChatGroq` before anything else in this plan is built on top of it.
If this fails, stop and re-check the model id against `console.groq.com/docs/model/openai/gpt-oss-120b` before proceeding to Task 2.

- [ ] **Step 6: Commit**

```bash
git add sentinel/llm/provider.py tests/test_provider.py
git commit -m "fix(agents): swap deprecated Groq models for their recommended replacements"
```

---

### Task 2: Add harness config settings

**Files:**
- Modify: `sentinel/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Settings.sentinel_model_call_thread_limit: int`, `Settings.sentinel_model_call_run_limit: int`, `Settings.sentinel_tool_call_thread_limit: int`, `Settings.sentinel_tool_call_run_limit: int`, `Settings.sentinel_retry_max_attempts: int` - consumed by Task 3 (limits) and Task 4 (retries).

- [ ] **Step 1: Write the failing test**

Check `tests/test_config.py`'s existing style first. Add:

```python
def test_harness_limit_settings_have_sane_defaults():
    from sentinel.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.sentinel_model_call_thread_limit == 40
    assert settings.sentinel_model_call_run_limit == 15
    assert settings.sentinel_tool_call_thread_limit == 40
    assert settings.sentinel_tool_call_run_limit == 20
    assert settings.sentinel_retry_max_attempts == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_harness_limit_settings_have_sane_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'sentinel_model_call_thread_limit'`.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/config.py`, add to the `Settings` class, after `sentinel_model_cheap`:

```python
    # Harness middleware: runaway-loop / cost insurance.
    sentinel_model_call_thread_limit: int = 40
    sentinel_model_call_run_limit: int = 15
    sentinel_tool_call_thread_limit: int = 40
    sentinel_tool_call_run_limit: int = 20

    # Harness middleware: retry attempts for a failed model call.
    sentinel_retry_max_attempts: int = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, all tests in the file green.

- [ ] **Step 5: Commit**

```bash
git add sentinel/config.py tests/test_config.py
git commit -m "feat(config): add harness middleware limit and retry settings"
```

---

### Task 3: The `corrective_feedback` helper

**Files:**
- Create: `sentinel/agents/harness.py`
- Test: Create `tests/test_harness.py`

**Interfaces:**
- Consumes: `sentinel.agents.registry.Registry` (has `.list() -> list[str]`, already exists), a `BaseChatModel` (`tools_chat_model`, already exists as a `build_agent` parameter).
- Produces: `make_corrective_feedback(tools_chat_model: BaseChatModel, registry: Registry) -> Callable[[Exception], str]` - consumed by Tasks 4 and 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness.py`:

```python
"""Tests for the harness's corrective-feedback message builder."""
from __future__ import annotations

import groq
import httpx
from langchain_core.messages import AIMessage

from sentinel.agents.harness import make_corrective_feedback
from tests.fakes import FakeChatModel


def _groq_tool_use_failed(missing: list[str]) -> groq.BadRequestError:
    """Build the exact exception shape Groq raises for a schema-invalid tool call."""
    fields = ", ".join(f"'{f}'" for f in missing)
    body = {
        "error": {
            "message": (
                "tool call validation failed: parameters for tool save_config "
                f"did not match schema: errors: [missing properties: {fields}]"
            ),
            "type": "invalid_request_error",
            "code": "tool_use_failed",
        }
    }
    response = httpx.Response(400, json=body, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.BadRequestError(str(body), response=response, body=body)


def _registry(tmp_path, models):
    from sentinel.agents.registry import Registry

    registry = Registry(tmp_path / "models")
    for model_id in models:
        (registry.root / model_id).mkdir(parents=True)
    manifest = registry._read_manifest()
    manifest["models"] = models
    registry._write_manifest(manifest)
    return registry


def test_tier1_names_missing_fields_from_groq_error(tmp_path):
    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        registry=registry,
    )
    error = _groq_tool_use_failed(
        ["framing", "failure_threshold", "reporting_cadence", "success_metric"]
    )
    message = feedback(error)
    assert "framing" in message
    assert "failure_threshold" in message
    assert "reporting_cadence" in message
    assert "success_metric" in message
    assert "ask the user" in message.lower()


def test_tier1_names_valid_ids_on_key_error(tmp_path):
    registry = _registry(tmp_path, ["et-v1", "et-v2"])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        registry=registry,
    )
    message = feedback(KeyError("et-v99"))
    assert "et-v99" in message
    assert "et-v1" in message
    assert "et-v2" in message


def test_tier2_falls_back_to_cheap_model_for_unrecognized_errors(tmp_path):
    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("The disk is full; try again shortly.")])
        ),
        registry=registry,
    )
    message = feedback(OSError("no space left on device"))
    assert message == "The disk is full; try again shortly."
```

(Three tests in this task; Task 6 appends a fourth to this same file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinel.agents.harness'`.

- [ ] **Step 3: Write minimal implementation**

Create `sentinel/agents/harness.py`:

```python
"""Custom harness middleware: corrective feedback, model-failure formatting,
and invalid-tool-call handling that LangChain's built-in middleware doesn't
cover.

Design: docs/superpowers/specs/2026-07-11-agent-harness-middleware.md
"""
from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .registry import Registry

_MISSING_PROPERTIES = re.compile(r"missing properties:\s*(.+?)\]")
_QUOTED = re.compile(r"'([^']+)'")


def _tier1_template(error: Exception, registry: Registry) -> str | None:
    """Return a deterministic corrective message, or None if the shape is unrecognized."""
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
    """Ask the cheap-tier model to summarize an unrecognized error in one sentence.

    Stateless: no chat history, just the raw error text.
    """
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
    """Build a corrective-feedback function for use as a middleware failure handler.

    Tries a deterministic template first (instant, free, exact); falls back to
    one stateless cheap-tier LLM call only for error shapes the templates
    don't recognize.
    """

    def corrective_feedback(error: Exception) -> str:
        templated = _tier1_template(error, registry)
        if templated is not None:
            return templated
        return _tier2_cheap_model(error, tools_chat_model)

    return corrective_feedback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_harness.py -v`
Expected: PASS, all three tests green.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/harness.py tests/test_harness.py
git commit -m "feat(agents): add corrective-feedback helper for the harness middleware stack"
```

---

### Task 4: Runaway-loop and cost insurance

**Files:**
- Modify: `sentinel/agents/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `get_settings().sentinel_model_call_thread_limit` etc. (Task 2), `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` (`langchain.agents.middleware`, verified signatures below).
- Produces: `build_agent(...)`'s `middleware=` list now includes these two - later tasks append to the same list, in the order specified in the design spec.

- [ ] **Step 1: Write the failing test**

In `tests/test_agent.py`, add (uses the existing `_build`/`_tc` helpers already in the file):

```python
def test_model_call_limit_ends_run_instead_of_looping_forever(tmp_path):
    """A model stuck calling a tool forever must stop at the configured limit, not hang."""
    import os

    from sentinel.config import get_settings

    os.environ["SENTINEL_MODEL_CALL_RUN_LIMIT"] = "3"
    get_settings.cache_clear()
    try:
        # Script more tool calls than the run limit allows - a "stuck" agent.
        scripted = [
            AIMessage(content="", tool_calls=[_tc("inspect", {"what": "registry"}, f"c{i}")])
            for i in range(10)
        ]
        agent = _build(tmp_path, scripted)
        thread = {"configurable": {"thread_id": "limit-test"}}
        final = agent.invoke(
            {"messages": [HumanMessage("loop forever")], "autonomy": "autonomous"},
            thread,
        )
        # The run ended (didn't hang/raise) at or near the configured limit,
        # not after exhausting all 10 scripted calls. ModelCallLimitMiddleware
        # itself appends one extra "limit reached" AIMessage when it ends the
        # run (exit_behavior="end"), so allow for that one extra message.
        model_messages = [m for m in final["messages"] if isinstance(m, AIMessage)]
        assert len(model_messages) <= 4
    finally:
        del os.environ["SENTINEL_MODEL_CALL_RUN_LIMIT"]
        get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_model_call_limit_ends_run_instead_of_looping_forever -v`
Expected: FAIL - all 10 scripted calls run, `len(model_messages) == 10`.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/agents/agent.py`, add the import and wire the middleware:

```python
from langchain.agents.middleware import AgentState, ModelCallLimitMiddleware, ToolCallLimitMiddleware
```

In `build_agent`, before the `return create_agent(...)`:

```python
    settings = get_settings()
    middleware = [
        ModelCallLimitMiddleware(
            thread_limit=settings.sentinel_model_call_thread_limit,
            run_limit=settings.sentinel_model_call_run_limit,
        ),
        ToolCallLimitMiddleware(
            thread_limit=settings.sentinel_tool_call_thread_limit,
            run_limit=settings.sentinel_tool_call_run_limit,
        ),
    ]
    return create_agent(
        chat_model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=DSAgentState,
        checkpointer=checkpointer,
        middleware=middleware,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS, including the new test and all pre-existing ones in the file.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/agent.py tests/test_agent.py
git commit -m "feat(agents): add model/tool call limit middleware to stop runaway loops"
```

---

### Task 5: The model-call resilience ladder (retry, fallback, graceful failure)

This is the direct fix for the `save_config` incident. Three layers, outermost to innermost: `ModelFailureFormatterMiddleware` (new, catches whatever survives the two layers below), `ModelFallbackMiddleware` (built-in, tries an alternate provider), `ModelRetryMiddleware` (built-in, retries the same model, `on_failure="error"` so failures propagate outward instead of being swallowed).

**Files:**
- Modify: `sentinel/agents/harness.py` (add `ModelFailureFormatterMiddleware`)
- Modify: `sentinel/agents/agent.py` (wire all three, add `_default_fallback_model` and a `fallback_chat_model` parameter)
- Modify: `tests/fakes.py` (add a chat model that raises on its first N calls)
- Modify: `tests/test_agent.py`'s `_build` helper, `tests/test_api.py`'s `_app`/`_slow_app` helpers, and `tests/test_cli.py`'s `build_agent` call (all three need an explicit `fallback_chat_model` from this task onward - see Step 4)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `make_corrective_feedback` (Task 3), `get_settings().sentinel_retry_max_attempts` (Task 2).
- Produces: `ModelFailureFormatterMiddleware(corrective_feedback: Callable[[Exception], str])` in `harness.py`; `build_agent(..., fallback_chat_model=None)` new optional parameter; `tests/fakes.py`'s `RaisingThenFakeChatModel`.

- [ ] **Step 1: Write the failing tests**

In `tests/fakes.py`, add (after `SlowFakeChatModel`):

```python
class RaisingThenFakeChatModel(FakeChatModel):
    """Raises a scripted exception on its first `fail_times` calls, then behaves normally."""

    fail_times: int = 1
    exception_factory: object = None  # Callable[[], Exception], set post-construction

    def _generate(self, *args, **kwargs):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.exception_factory()
        return super()._generate(*args, **kwargs)
```

In `tests/test_agent.py`, add (reuse `_groq_tool_use_failed` from `tests/test_harness.py` - import it):

```python
from tests.fakes import RaisingThenFakeChatModel
from tests.test_harness import _groq_tool_use_failed


def test_model_call_failure_ends_gracefully_instead_of_crashing(tmp_path):
    """The save_config incident: a malformed tool call rejected by the provider
    must produce a graceful message, not a crashed run."""
    from langgraph.checkpoint.memory import MemorySaver

    from sentinel.agents.agent import build_agent

    failing_model = RaisingThenFakeChatModel(
        messages=iter([AIMessage("unreachable - always raises in this test")]),
        fail_times=10,  # more than max_retries + fallback attempts, so it never recovers
    )
    failing_model.exception_factory = lambda: _groq_tool_use_failed(
        ["framing", "failure_threshold", "reporting_cadence", "success_metric"]
    )
    agent = build_agent(
        chat_model=failing_model,
        train_fn=lambda cfg: (_ for _ in ()).throw(AssertionError("not reached")),
        retrain_fn=lambda *a: (_ for _ in ()).throw(AssertionError("not reached")),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
        fallback_chat_model=failing_model,  # fallback also fails, forcing the formatter path
    )
    thread = {"configurable": {"thread_id": "incident"}}
    final = agent.invoke(
        {"messages": [HumanMessage("use sensible defaults")], "autonomy": "guarded"},
        thread,
    )
    last = final["messages"][-1]
    assert "framing" in last.content
    assert "failure_threshold" in last.content
    assert "BadRequestError" not in last.content
    assert "Traceback" not in last.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_model_call_failure_ends_gracefully_instead_of_crashing -v`
Expected: FAIL - either `TypeError: build_agent() got an unexpected keyword argument 'fallback_chat_model'`, or (once that's stubbed) the real `groq.BadRequestError` propagating out of `agent.invoke(...)` uncaught.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/agents/harness.py`, add:

```python
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage


class ModelFailureFormatterMiddleware(AgentMiddleware):
    """Outermost model-call safety net: catches whatever retry and fallback
    couldn't fix, and turns it into a graceful reply instead of a crash.
    """

    def __init__(self, corrective_feedback: Callable[[Exception], str]) -> None:
        super().__init__()
        self._corrective_feedback = corrective_feedback

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as error:  # noqa: BLE001 - intentionally broad, this is the last resort
            return AIMessage(content=self._corrective_feedback(error))

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:  # noqa: BLE001 - intentionally broad, this is the last resort
            return AIMessage(content=self._corrective_feedback(error))
```

In `sentinel/agents/agent.py`, add imports:

```python
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware

from ..llm.provider import get_chat_model
from .harness import ModelFailureFormatterMiddleware, make_corrective_feedback
```

Add a default-fallback-model helper, mirroring `_default_checkpointer`:

```python
def _default_fallback_model():
    """The provider Sentinel doesn't call by default, used as a model-call fallback."""
    return get_chat_model("smart", name="anthropic")
```

Update `build_agent`'s signature and body:

```python
def build_agent(
    *,
    chat_model,
    train_fn,
    retrain_fn,
    tools_chat_model,
    ticket_dir,
    models_dir,
    checkpointer=None,
    fallback_chat_model=None,
):
    """Assemble the registry, tools, and create_agent hub."""
    registry = Registry(models_dir)
    tools = make_tools(
        train_fn=train_fn,
        retrain_fn=retrain_fn,
        chat_model=tools_chat_model,
        ticket_dir=ticket_dir,
        registry=registry,
    )
    if checkpointer is None:
        checkpointer = _default_checkpointer()
    if fallback_chat_model is None:
        fallback_chat_model = _default_fallback_model()

    settings = get_settings()
    corrective_feedback = make_corrective_feedback(tools_chat_model, registry)
    middleware = [
        ModelCallLimitMiddleware(
            thread_limit=settings.sentinel_model_call_thread_limit,
            run_limit=settings.sentinel_model_call_run_limit,
        ),
        ToolCallLimitMiddleware(
            thread_limit=settings.sentinel_tool_call_thread_limit,
            run_limit=settings.sentinel_tool_call_run_limit,
        ),
        ModelFailureFormatterMiddleware(corrective_feedback),
        ModelFallbackMiddleware(fallback_chat_model),
        ModelRetryMiddleware(
            max_retries=settings.sentinel_retry_max_attempts,
            on_failure="error",
        ),
    ]
    return create_agent(
        chat_model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=DSAgentState,
        checkpointer=checkpointer,
        middleware=middleware,
    )
```

- [ ] **Step 4: Fix the other test helpers before they break**

`_default_fallback_model()` constructs a real `ChatAnthropic`, which raises `pydantic.ValidationError` immediately if `ANTHROPIC_API_KEY` isn't set (verified: `ChatAnthropic(api_key=None, ...)` fails at construction, not at call time).
Every other test file's `build_agent(...)` call site that doesn't pass `fallback_chat_model` explicitly would now crash in the normal no-live-key test environment - this must be fixed in this same task, not left for later, since Tasks 6-11 all reuse these same helpers. Codex's review caught a third call site (`tests/test_cli.py`) that an earlier pass of this plan missed.

In `tests/test_agent.py`, update `_build`:

```python
def _build(tmp_path, scripted):
    from sentinel.agents.agent import build_agent

    return build_agent(
        chat_model=FakeChatModel(messages=iter(scripted)),
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda mid, hp, rc, window: _fake_training_run(
            tmp_path, rmse=16.0
        ),
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("Report body.")])
        ),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
        fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
    )
```

In `tests/test_api.py`, update both `_app`'s and `_slow_app`'s inner `factory(checkpointer)` functions to add the same `fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")]))` keyword argument to their `build_agent(...)` calls.

In `tests/test_cli.py`'s `test_run_turn_completes_without_interrupt_in_autonomous`, add the same keyword argument to its `build_agent(...)` call.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py tests/test_api.py tests/test_cli.py -v`
Expected: PASS, including the new incident-shaped test and every pre-existing test in all three files - this is the check that proves the fallback-model fix above actually prevented the regression.
Note the retry/fallback attempts add real (short) delays in the incident-shaped test - `ModelRetryMiddleware`'s default backoff is small but non-zero; if the test is slow, that's expected, not a bug.

- [ ] **Step 6: Commit**

```bash
git add sentinel/agents/harness.py sentinel/agents/agent.py tests/fakes.py tests/test_agent.py tests/test_api.py tests/test_cli.py
git commit -m "feat(agents): add model-call retry, fallback, and graceful-failure middleware"
```

---

### Task 6: `InvalidToolCallMiddleware` - close the GH #33504 gap

**Files:**
- Modify: `sentinel/agents/harness.py`
- Modify: `sentinel/agents/agent.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `corrective_feedback` (Task 3).
- Produces: `InvalidToolCallMiddleware(corrective_feedback: Callable[[Exception], str])` in `harness.py`, appended to `build_agent`'s `middleware` list.

Simpler than the first draft of this plan: instead of synthesizing a matching `tool_calls` entry alongside the corrective `ToolMessage`, this uses `create_agent`'s explicit `jump_to` control-flow field (`AgentState`'s built-in `jump_to: NotRequired[JumpTo | None]` channel, verified present in the installed source) - routing checks for an explicit `jump_to` before it even looks at `tool_calls`, so `after_model` can send execution straight back to the model without touching the AI message at all.

- [ ] **Step 1: Write the failing test**

In `tests/test_harness.py`, add:

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from sentinel.agents.harness import InvalidToolCallMiddleware


class _InvalidToolCallChatModel(FakeChatModel):
    """A fake whose FIRST response has malformed JSON args LangChain can't parse.

    Every later call behaves normally, so the test can assert the loop
    actually continued past the corrective ToolMessage instead of just
    checking the corruption happened.
    """

    already_corrupted: bool = False

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        if self.already_corrupted:
            return result
        self.already_corrupted = True
        message = result.generations[0].message
        message.tool_calls = []
        message.invalid_tool_calls = [
            {
                "type": "invalid_tool_call",
                "id": "bad1",
                "name": "save_config",
                "args": "{not valid json",
                "error": "Invalid JSON",
            }
        ]
        return result


def test_invalid_tool_call_gets_a_corrective_message_and_the_loop_continues(tmp_path):
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentState

    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        registry=registry,
    )

    @tool
    def save_config(x: int) -> str:
        """A stand-in tool."""
        return "saved"

    scripted = [
        AIMessage(content="calling save_config"),  # gets its tool_calls overwritten by the fake above
        AIMessage(content="Understood, let me ask for the missing fields."),
    ]
    model = _InvalidToolCallChatModel(messages=iter(scripted))
    agent = create_agent(
        model,
        [save_config],
        state_schema=AgentState,
        checkpointer=MemorySaver(),
        middleware=[InvalidToolCallMiddleware(feedback)],
    )
    final = agent.invoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "invalid-tc"}},
    )
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "bad1"
    assert "Invalid JSON" in tool_messages[0].content or "invalid" in tool_messages[0].content.lower()
    # The loop continued past the malformed call to the second scripted message.
    assert "Understood" in final["messages"][-1].content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_harness.py::test_invalid_tool_call_gets_a_corrective_message_and_the_loop_continues -v`
Expected: FAIL with `ImportError: cannot import name 'InvalidToolCallMiddleware'`.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/agents/harness.py`, add:

```python
from langchain_core.messages import ToolMessage


class InvalidToolCallMiddleware(AgentMiddleware):
    """Closes a real LangChain gap (issue #33504): when the model's JSON is too
    malformed to parse into `tool_calls`, it lands in `invalid_tool_calls`
    instead, and the default routing silently ends the turn without ever
    telling the model what went wrong.

    Uses `create_agent`'s explicit `jump_to="model"` control-flow field
    (checked by routing before it looks at `tool_calls` at all) to send
    execution straight back to the model with the correction already in
    context - no need to synthesize a fake `tool_calls` entry.
    """

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

        tool_messages: list[ToolMessage] = []
        for invalid in last.invalid_tool_calls:
            call_id = invalid.get("id") or "unknown"
            name = invalid.get("name") or "unknown_tool"
            error = ValueError(invalid.get("error") or "the tool call could not be parsed")
            tool_messages.append(
                ToolMessage(
                    content=self._corrective_feedback(error),
                    tool_call_id=call_id,
                    name=name,
                    status="error",
                )
            )

        return {"messages": tool_messages, "jump_to": "model"}
```

In `sentinel/agents/agent.py`, add the import:

```python
from .harness import InvalidToolCallMiddleware, ModelFailureFormatterMiddleware, make_corrective_feedback
```

(replacing the narrower import from Task 5). Append to the `middleware` list, right after `ModelRetryMiddleware(...)`:

```python
        InvalidToolCallMiddleware(corrective_feedback),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_harness.py tests/test_agent.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/harness.py sentinel/agents/agent.py tests/test_harness.py
git commit -m "feat(agents): add InvalidToolCallMiddleware to close a LangChain routing gap"
```

---

### Task 7: Replace `confirm()`/`interrupt()` with `HumanInTheLoopMiddleware` (backend)

Backend only - `sentinel/agents/tools.py` and `sentinel/agents/agent.py`. Task 8 adapts the API, Task 9 adapts the frontend, Task 10 adapts the CLI.

**Files:**
- Modify: `sentinel/agents/tools.py` (remove `confirm()` and its five call sites)
- Modify: `sentinel/agents/harness.py` (add the `guarded_when` factory)
- Modify: `sentinel/agents/agent.py` (wire `HumanInTheLoopMiddleware`)
- Test: `tests/test_agent.py` (rewrite the guarded-confirmation test, add a genuinely batched test)

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: guarded tools (`train`, `retrain`, `promote`, `delete`, `run_monitor`) no longer take an `Annotated[dict, InjectedState]` `state` parameter or call `confirm()`. `state.interrupts[0].value` is now shaped `{"action_requests": [...], "review_configs": [...]}` instead of `{"type": "confirm", "tool": ..., "detail": ...}`, and there is exactly **one** interrupt per pause (never more), bundling every simultaneously-pending guarded call - Task 8's API adapter depends on this exact shape.

- [ ] **Step 1: Write the failing tests**

Replace the existing `test_guarded_confirmation_interrupt_and_mapped_resume` in `tests/test_agent.py` with:

```python
def test_guarded_confirmation_interrupt_and_mapped_resume(tmp_path):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "t3"}}
    agent.invoke(
        {"messages": [HumanMessage("train")], "autonomy": "guarded"},
        thread,
    )
    state = agent.get_state(thread)
    assert state.interrupts
    request = state.interrupts[0].value
    assert request["action_requests"][0]["name"] == "train"
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), thread
    )
    assert "Trained" in final["messages"][-1].content


def test_batched_confirmation_two_guarded_calls_in_one_message(tmp_path):
    """Two guarded tool calls in the SAME AI message (real batching, not two
    sequential turns) must produce one interrupt bundling both, resolved
    together by one ordered resume."""
    from sentinel.agents.registry import Registry

    models_dir = tmp_path / "models"
    registry = Registry(models_dir)
    manifest = registry._read_manifest()
    manifest["models"] = ["et-v1", "et-v2"]
    manifest["active"] = "et-v1"
    registry._write_manifest(manifest)
    for model_id in ("et-v1", "et-v2"):
        (registry.root / model_id).mkdir()
        (registry.root / model_id / "metrics.json").write_text(
            '{"rmse": 17.1, "mae": 12.0, "r2": 0.83}'
        )
        (registry.root / model_id / "provenance.json").write_text("{}")

    # promote(et-v1) is a no-op reassignment (et-v1 is already active) and
    # delete(et-v2) is a non-active model - deliberately order-independent,
    # so this test doesn't depend on which of the two runs first.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                _tc("promote", {"model_id": "et-v1"}, "c1"),
                _tc("delete", {"model_id": "et-v2"}, "c2"),
            ],
        ),
        AIMessage(content="Confirmed both actions."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "batched"}}
    agent.invoke(
        {
            "messages": [HumanMessage("promote et-v1 and delete et-v2")],
            "autonomy": "guarded",
        },
        thread,
    )
    state = agent.get_state(thread)
    assert len(state.interrupts) == 1
    request = state.interrupts[0].value
    assert {a["name"] for a in request["action_requests"]} == {"promote", "delete"}
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}),
        thread,
    )
    assert "Confirmed both actions" in final["messages"][-1].content
    assert registry.active() == "et-v1"
    assert registry.list() == ["et-v1"]


def test_autonomous_mode_still_skips_confirmation(tmp_path):
    """Autonomy check must survive the swap to HumanInTheLoopMiddleware."""
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "auto"}}
    final = agent.invoke(
        {"messages": [HumanMessage("train")], "autonomy": "autonomous"},
        thread,
    )
    state = agent.get_state(thread)
    assert not state.interrupts
    assert "Trained" in final["messages"][-1].content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -k "confirmation or batched or autonomous_mode" -v`
Expected: FAIL - `state.interrupts[0].value` is still `{"type": "confirm", ...}`, no `action_requests` key.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/agents/tools.py`, delete the `confirm()` function entirely (lines ~36-51). Check whether `interrupt` (from `langgraph.types`) and `get_stream_writer` (from `langgraph.config`) have any remaining uses in the file once `confirm()` is gone (`grep -n "interrupt\|get_stream_writer" sentinel/agents/tools.py`) - if not, remove those imports too.

In each of the five guarded tools, remove the `state: Annotated[dict, InjectedState]` parameter and the `confirm(...)` / `if denial: return denial` block. For example, `promote` becomes:

```python
    @tool
    def promote(model_id: str) -> str:
        """Set a registered model as active."""
        try:
            registry.set_active(model_id)
        except KeyError:
            return _unknown(model_id)
        return f"Promoted {model_id}; it is now the active model."
```

Apply the same pattern to `delete`, `run_monitor`, `train`, and `retrain` - remove the `state` parameter and the `confirm(...)` block, keep everything else in each tool body unchanged. If `InjectedState` has no remaining uses in the file after this, remove its import too (`grep -n InjectedState sentinel/agents/tools.py`).

In `sentinel/agents/harness.py`, add:

```python
def guarded_when(tool_name: str):
    """Build a `when` predicate for `HumanInTheLoopMiddleware.interrupt_on`.

    Returns False (skip the interrupt, auto-approve) when the session's
    autonomy is "autonomous" - streaming the same `auto_approved` notice the
    old hand-rolled `confirm()` used to, including the tool call's arguments,
    so autonomous runs stay auditable.
    Returns True (raise the interrupt) otherwise.
    """

    def when(request) -> bool:
        if request.state.get("autonomy") == "autonomous":
            request.runtime.stream_writer(
                {
                    "type": "auto_approved",
                    "tool": tool_name,
                    "detail": str(request.tool_call["args"]),
                }
            )
            return False
        return True

    return when
```

In `sentinel/agents/agent.py`, add the import:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

from .harness import guarded_when
```

Append to the `middleware` list, as the last (innermost) entry:

```python
        HumanInTheLoopMiddleware(
            interrupt_on={
                name: {"allowed_decisions": ["approve", "reject"], "when": guarded_when(name)}
                for name in ("train", "retrain", "promote", "delete", "run_monitor")
            },
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS, all tests in the file green, including the rewritten and new ones.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/tools.py sentinel/agents/harness.py sentinel/agents/agent.py tests/test_agent.py
git commit -m "feat(agents): replace hand-rolled confirm()/interrupt() with HumanInTheLoopMiddleware"
```

---

### Task 8: Adapt the API (resume + snapshot) to the new bundled confirmation shape

**Files:**
- Modify: `sentinel/api/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `state.interrupts[0].value == {"action_requests": [{"name", "args", "description"?}, ...], "review_configs": [...]}` (Task 7). There is always exactly one interrupt at a time.
- Produces: `POST /sessions/{id}/resume` now requires **all** answers for the currently-pending confirmations in a single call (matching what `HumanInTheLoopMiddleware` itself requires) - Task 9's frontend change is what makes this true in practice. `GET /sessions/{id}`'s `pending_confirmations` list now has the same per-card `{interrupt, tool, detail}` shape as before, expanded from the bundled interrupt.

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`'s existing `test_guarded_session_emits_confirm_with_interrupt_id` (around line 183) doesn't hardcode the interrupt id's format, so it should keep passing unchanged through this task - it's the regression check that the SSE `confirm` event contract didn't break.

The shared `_app(tmp_path)` helper always scripts a single `train` call - it's reused by many other tests in this file and must not change. The two new tests below need a *different* script (two guarded tool calls in one message), so add a dedicated builder alongside `_app`/`_slow_app` rather than touching the shared one:

```python
def _batched_guarded_app(tmp_path):
    """An app whose one scripted turn requests two guarded tool calls at once."""
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=FakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "promote", "args": {"model_id": "et-v1"}, "id": "c1"},
                                {"name": "delete", "args": {"model_id": "et-v2"}, "id": "c2"},
                            ],
                        ),
                        AIMessage(content="Confirmed both actions."),
                    ]
                )
            ),
            train_fn=lambda cfg: _fake_run(tmp_path),
            retrain_fn=lambda *args: _fake_run(tmp_path),
            tools_chat_model=FakeChatModel(messages=iter([AIMessage("report")])),
            ticket_dir=str(tmp_path / "tickets"),
            models_dir=str(tmp_path / "models"),
            checkpointer=checkpointer,
            fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
        )

    return create_app(
        agent_factory=factory,
        checkpointer=MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def _seed_two_models(tmp_path):
    from sentinel.agents.registry import Registry

    registry = Registry(tmp_path / "models")
    manifest = registry._read_manifest()
    manifest["models"] = ["et-v1", "et-v2"]
    manifest["active"] = "et-v1"
    registry._write_manifest(manifest)
    for model_id in ("et-v1", "et-v2"):
        (registry.root / model_id).mkdir()
        (registry.root / model_id / "metrics.json").write_text(
            '{"rmse": 17.1, "mae": 12.0, "r2": 0.83}'
        )
        (registry.root / model_id / "provenance.json").write_text("{}")
    return registry
```

Add the tests, using these two helpers:

```python
def test_resume_rejects_partial_answers_for_a_batched_confirmation(tmp_path):
    """HumanInTheLoopMiddleware requires every bundled decision at once - the
    API must reject an incomplete answer set with a clear error, not silently
    misroute it."""
    _seed_two_models(tmp_path)
    app = _batched_guarded_app(tmp_path)
    thread_id = _start_session(app, message="promote et-v1 and delete et-v2", autonomy="guarded")
    snap = _request(app, "GET", f"/sessions/{thread_id}").json()
    assert len(snap["pending_confirmations"]) == 2
    first_id = snap["pending_confirmations"][0]["interrupt"]

    response = _request(
        app, "POST", f"/sessions/{thread_id}/resume",
        json={"answers": {first_id: "yes"}},  # only one of two - incomplete
    )
    assert response.status_code == 400


def test_snapshot_expands_bundled_confirmations_independently(tmp_path):
    _seed_two_models(tmp_path)
    app = _batched_guarded_app(tmp_path)
    thread_id = _start_session(app, message="promote et-v1 and delete et-v2", autonomy="guarded")
    snap = _request(app, "GET", f"/sessions/{thread_id}").json()
    cards = snap["pending_confirmations"]
    assert len(cards) == 2
    assert {c["tool"] for c in cards} == {"promote", "delete"}
    for card in cards:
        assert set(card) == {"interrupt", "tool", "detail"}
```

Check `_start_session`'s exact existing signature first (`grep -n "_start_session" -A5 tests/test_api.py`) - it already accepts `(app, message="train", autonomy="guarded")`, so no change needed there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k "resume_rejects or snapshot_expands" -v`
Expected: FAIL - `_stream()` and `snapshot()` still read `state.interrupts` assuming the old per-tool value shape and don't validate completeness.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/api/app.py`, replace the interrupt-to-SSE emission, the `resume` route, and the `snapshot` route.

First, the emission side inside `_stream()` - this replaces the existing block at the same indentation, still nested inside the function's `async with lock:` (12-space indent, matching the surrounding `state = await asyncio.to_thread(agent.get_state, thread)` line it's replacing):

```python
            state = await asyncio.to_thread(agent.get_state, thread)
            if state.interrupts:
                request = state.interrupts[0].value
                interrupt_id = state.interrupts[0].id
                for index, action in enumerate(request["action_requests"]):
                    yield _sse(
                        "confirm",
                        {
                            "interrupt": f"{interrupt_id}:{index}",
                            "tool": action["name"],
                            "detail": json.dumps(action["args"]),
                        },
                    )
            else:
                yield _sse("done", {})
```

Add a small shared helper near the other module-level helpers (`_sse`, `_transcript`) - both `resume` and `snapshot` need to turn an interrupt's `action_requests` into the same per-card shape:

```python
def _pending_cards(state) -> list[dict]:
    if not state.interrupts:
        return []
    request = state.interrupts[0].value
    interrupt_id = state.interrupts[0].id
    return [
        {
            "interrupt": f"{interrupt_id}:{index}",
            "tool": action["name"],
            "detail": json.dumps(action["args"]),
        }
        for index, action in enumerate(request["action_requests"])
    ]
```

Replace `resume`'s body:

```python
    @app.post("/sessions/{thread_id}/resume")
    async def resume(thread_id: str, body: dict):
        thread = _thread(thread_id)
        await _require(thread)
        state = await asyncio.to_thread(agent.get_state, thread)
        if not state.interrupts:
            raise HTTPException(status_code=400, detail="no confirmation is pending")
        request = state.interrupts[0].value
        interrupt_id = state.interrupts[0].id
        expected = len(request["action_requests"])
        answers = body.get("answers")
        if answers is None:
            if expected != 1:
                raise HTTPException(
                    status_code=400,
                    detail="multiple confirmations pending; use 'answers' map",
                )
            answers = {f"{interrupt_id}:0": body.get("answer", "")}
        by_index: dict[int, str] = {}
        for composite_id, answer in answers.items():
            base, _, index = composite_id.rpartition(":")
            if base != interrupt_id or not index.isdigit():
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown confirmation id {composite_id!r}",
                )
            by_index[int(index)] = answer
        if sorted(by_index) != list(range(expected)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"expected answers for all {expected} pending confirmation(s), "
                    f"got {sorted(by_index)}"
                ),
            )
        decisions = [
            {"type": "approve"} if by_index[i].strip().lower() in {"y", "yes"} else {"type": "reject"}
            for i in range(expected)
        ]
        return StreamingResponse(
            _stream(Command(resume={"decisions": decisions}), thread),
            media_type="text/event-stream",
            headers={"x-thread-id": thread_id},
        )
```

Replace `snapshot`'s `pending_confirmations` construction:

```python
    @app.get("/sessions/{thread_id}")
    async def snapshot(thread_id: str):
        thread = _thread(thread_id)
        await _require(thread)
        state = await asyncio.to_thread(agent.get_state, thread)
        messages = state.values.get("messages", [])
        return {
            "autonomy": state.values.get("autonomy"),
            "pending_confirmations": _pending_cards(state),
            "last_message": (
                messages[-1].content if messages else None
            ),
            "messages": _transcript(messages),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS, all tests in the file green, including the pre-existing guarded-session test (unchanged assertions) and the two new ones.

- [ ] **Step 5: Commit**

```bash
git add sentinel/api/app.py tests/test_api.py
git commit -m "feat(api): adapt resume and snapshot endpoints to HumanInTheLoopMiddleware's bundled confirmations"
```

---

### Task 9: Batch confirmation submission in the frontend

This is a real, visible UX change, not a bug fix: today each `ConfirmCard` resolves the instant you click Yes/No. After this task, clicking records a local choice, and a "Submit responses" action sends every currently-pending card's decision together in one request - matching what `HumanInTheLoopMiddleware` requires server-side (Task 8).

**Files:**
- Modify: `web/src/state/useSession.ts` (add `decision` to `Pending`, add `choose`/`resolveAll` actions, remove `resolve`)
- Modify: `web/src/components/ConfirmCard.tsx` (local choice instead of immediate submission)
- Modify: `web/src/components/ChatView.tsx` (batch submission)
- Test: `web/src/state/useSession.test.ts`

**Interfaces:**
- Consumes: nothing new from backend tasks - this only depends on the already-shipped `POST /sessions/{id}/resume` accepting `{"answers": {"<id>": "yes"|"no", ...}}` for multiple ids at once (already true; Task 8 made it strict about requiring *all* of them).
- Produces: nothing consumed by later tasks in this plan.

- [ ] **Step 1: Write the failing test**

Check `web/src/state/useSession.test.ts`'s existing style first (`cat web/src/state/useSession.test.ts`). Add:

```typescript
import { describe, expect, it } from "vitest";
import { initialSession, sessionReducer } from "./useSession";

describe("choose/resolveAll", () => {
  it("records a local decision without removing the pending card", () => {
    const withPending = sessionReducer(initialSession, {
      type: "event",
      event: { event: "confirm", data: { interrupt: "abc:0", tool: "train", detail: "{}" } },
    });
    const chosen = sessionReducer(withPending, {
      type: "choose",
      interrupt: "abc:0",
      decision: "yes",
    });
    expect(chosen.pending).toHaveLength(1);
    expect(chosen.pending[0].decision).toBe("yes");
  });

  it("resolveAll clears every pending card regardless of decision state", () => {
    const withPending = sessionReducer(initialSession, {
      type: "event",
      event: { event: "confirm", data: { interrupt: "abc:0", tool: "train", detail: "{}" } },
    });
    const cleared = sessionReducer(withPending, { type: "resolveAll" });
    expect(cleared.pending).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run useSession.test.ts`
Expected: FAIL - `"choose"` and `"resolveAll"` aren't valid `SessionAction` variants yet (TypeScript compile error surfaced through vitest), and `Pending` has no `decision` field.

- [ ] **Step 3: Write minimal implementation**

In `web/src/state/useSession.ts`, update the `Pending` interface, `SessionAction` union, reducer cases, and the `confirm` event handler:

```typescript
export interface Pending {
  interrupt: string;
  tool: string;
  detail: string;
  decision: "yes" | "no" | null;
}

export type SessionAction =
  | { type: "start" }
  | { type: "reset" }
  | { type: "user"; text: string }
  | { type: "event"; event: SentinelEvent }
  | { type: "hydrate"; messages: TranscriptMessage[] }
  | { type: "setPending"; pending: Pending[] }
  | { type: "choose"; interrupt: string; decision: "yes" | "no" }
  | { type: "resolveAll" };
```

Replace the `case "resolve":` branch in `sessionReducer` with:

```typescript
    case "choose":
      return {
        ...state,
        pending: state.pending.map((p) =>
          p.interrupt === action.interrupt ? { ...p, decision: action.decision } : p,
        ),
      };
    case "resolveAll":
      return { ...state, pending: [] };
```

In `applyEvent`'s `"confirm"` case, add `decision: null` to the pushed `Pending`:

```typescript
    case "confirm": {
      // Terminal: emitted in place of `done`, so also clear streaming.
      const d = ev.data as { tool: string; detail: string; interrupt: string };
      return {
        ...state,
        streaming: false,
        pending: [
          ...state.pending,
          { interrupt: d.interrupt, tool: d.tool, detail: d.detail, decision: null },
        ],
      };
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run useSession.test.ts`
Expected: PASS, both new tests green.

- [ ] **Step 5: Update `ConfirmCard.tsx` and `ChatView.tsx`, then run the full frontend check**

`web/src/components/ConfirmCard.tsx`, full replacement:

```tsx
import type { Pending } from "../state/useSession";

interface ConfirmCardProps {
  pending: Pending;
  onChoose: (value: "yes" | "no") => void;
  disabled: boolean;
}

export function ConfirmCard({ pending, onChoose, disabled }: ConfirmCardProps) {
  return (
    <div className="confirm">
      <div>
        <strong>{pending.tool}</strong>
        <p>{pending.detail}</p>
      </div>
      <div className="confirm-actions">
        <button
          type="button"
          onClick={() => onChoose("yes")}
          disabled={disabled}
          aria-pressed={pending.decision === "yes"}
          className={pending.decision === "yes" ? "chosen" : undefined}
        >
          Yes
        </button>
        <button
          type="button"
          onClick={() => onChoose("no")}
          disabled={disabled}
          aria-pressed={pending.decision === "no"}
          className={pending.decision === "no" ? "chosen" : undefined}
        >
          No
        </button>
      </div>
    </div>
  );
}
```

In `web/src/components/ChatView.tsx`, replace the `answer` function with a `choose` function plus a `submitDecisions` function, and update the JSX:

```tsx
  function choose(interrupt: string, value: "yes" | "no") {
    if (resumeBusy) return;
    dispatch({ type: "choose", interrupt, decision: value });
  }

  async function submitDecisions() {
    if (!threadId || resumeBusy) return;
    if (state.pending.length === 0 || !state.pending.every((p) => p.decision !== null)) return;
    setResumeBusy(true);
    const answers: Record<string, string> = {};
    for (const p of state.pending) {
      answers[p.interrupt] = p.decision as string;
    }
    const controller = new AbortController();
    streamAbort.current = controller;
    dispatch({ type: "resolveAll" });
    try {
      const response = await client.resume(threadId, answers, controller.signal);
      await consumeStream(response, dispatch);
      streamAbort.current = null;
      onTurnFinished();
    } finally {
      setResumeBusy(false);
    }
  }
```

Update the render section (replacing the existing `state.pending.map` block and adding a submit button after it):

```tsx
        {state.pending.map((p) => (
          <ConfirmCard
            key={p.interrupt}
            pending={p}
            disabled={resumeBusy}
            onChoose={(value) => choose(p.interrupt, value)}
          />
        ))}
        {state.pending.length > 0 && (
          <button
            type="button"
            onClick={submitDecisions}
            disabled={resumeBusy || !state.pending.every((p) => p.decision !== null)}
          >
            Submit responses
          </button>
        )}
```

Run: `cd web && npx tsc --noEmit && npx vitest run`
Expected: clean typecheck (confirms `ConfirmCard`'s new prop names and `ChatView`'s new functions line up with no type errors), all vitest tests pass.

- [ ] **Step 6: Commit**

```bash
git add web/src/state/useSession.ts web/src/state/useSession.test.ts web/src/components/ConfirmCard.tsx web/src/components/ChatView.tsx
git commit -m "feat(web): batch confirmation submission to match HumanInTheLoopMiddleware"
```

---

### Task 10: Migrate the CLI

**Files:**
- Modify: `sentinel/agents/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `state.interrupts[0].value == {"action_requests": [...], ...}` (Task 7).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, add:

```python
def test_run_turn_reports_pending_on_guarded_confirmation(tmp_path, monkeypatch):
    from sentinel.agents.__main__ import run_turn
    from sentinel.agents.agent import build_agent

    agent = build_agent(
        chat_model=FakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "train", "args": {}, "id": "c1"}],
                    ),
                    AIMessage(content="Trained et-v1."),
                ]
            )
        ),
        train_fn=lambda cfg: _fake_run(tmp_path),
        retrain_fn=lambda *args: _fake_run(tmp_path),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("report")])),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
        fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
    )
    thread = {"configurable": {"thread_id": "cli-guarded"}}
    output = []
    pending = run_turn(
        agent,
        thread,
        {"messages": [HumanMessage("train")], "autonomy": "guarded"},
        output.append,
    )
    assert pending is True
    state = agent.get_state(thread)
    assert state.interrupts[0].value["action_requests"][0]["name"] == "train"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_run_turn_reports_pending_on_guarded_confirmation -v`
Expected: FAIL or ERROR - the guarded run either raises (old code reading a shape that no longer exists) or the assertion on `action_requests` fails.

- [ ] **Step 3: Write minimal implementation**

In `sentinel/agents/__main__.py`, replace the resume loop inside `main()`:

```python
        pending = run_turn(agent, thread, graph_input)
        while pending:
            state = agent.get_state(thread)
            request = state.interrupts[0].value
            decisions = []
            for action in request["action_requests"]:
                answer = input(
                    f"Confirm {action['name']} ({action['args']})? [y/N] "
                )
                if answer.strip().lower() in {"y", "yes"}:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({"type": "reject"})
            pending = run_turn(
                agent,
                thread,
                Command(resume={"decisions": decisions}),
            )
```

This replaces the block from `pending = run_turn(agent, thread, graph_input)` through the end of the `while pending:` loop - the rest of `main()` (argument parsing, the outer `while True:` input loop, `first`/`graph_input` handling) is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, both tests in the file green.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/__main__.py tests/test_cli.py
git commit -m "feat(cli): migrate confirmation loop to HumanInTheLoopMiddleware's bundled decisions"
```

---

### Task 11: Full regression and live verification

**Files:** none (verification only).

**Interfaces:** none - this task only runs and observes.

- [ ] **Step 1: Full offline suite**

Run: `uv run pytest -q`
Expected: every test passes, including all tests added in Tasks 1-10 plus the full pre-existing suite (concurrency/deadlock tests from the earlier SSE-blocking fix, DS-core tests, everything).

- [ ] **Step 2: Frontend**

Run: `cd web && npx tsc --noEmit && npx vitest run`
Expected: clean typecheck, all vitest tests pass.

- [ ] **Step 3: Live smoke test against the real backend**

Start the real server (`GROQ_API_KEY` set in `.env`):

```bash
uv run uvicorn "sentinel.api.app:create_app" --factory --port 8000
```

In another terminal, reproduce the actual incident and confirm it no longer crashes:

```bash
curl -sN -X POST http://127.0.0.1:8000/sessions \
  -H "content-type: application/json" \
  -d '{"message": "give me the options"}'
```

Expected: a `message` SSE event with plain, readable text (no `BadRequestError`, no `Traceback`, no raw JSON error dump) - this is the actual bug from the start of this conversation, now closed at the source.

Then, with the frontend running (`npm run dev` in `web/`), drive a guarded flow end to end in the real UI: start a session, let the agent propose two guarded tool calls in one turn if you can prompt for it (e.g. "train, then promote whatever wins"), confirm that both cards render, that clicking Yes/No on each records a visible selection without submitting, that "Submit responses" is disabled until both are chosen, and that submitting resolves both together. Also re-check the autonomy toggle and leaderboard behave exactly as they did before this plan (no regression in the concurrency fixes from the earlier session).

- [ ] **Step 4: Update the harness field notes**

In `docs/learning/05-harness-engineering.html`, update the failure-map table's "Sentinel today" column for the rows now covered (model call failure, invalid tool-call parsing, runaway loops/cost, safety/consent) from their current `none`/`partial`/`handled`-with-caveats chips to reflect the new middleware coverage, mark tool execution failure as "not achievable via create_agent, see spec" rather than the original "handled" chip, and add one line under the existing "Concrete finding" callout noting the model swap shipped. Republish via the `Artifact` tool at the same file path so the existing shared link updates in place.

- [ ] **Step 5: Final commit**

```bash
git add docs/learning/05-harness-engineering.html
git commit -m "docs: update harness field notes to reflect the shipped middleware stack"
```
