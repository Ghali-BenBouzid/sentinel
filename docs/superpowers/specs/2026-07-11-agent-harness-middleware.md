# Agent harness middleware (design)

This is the design for Sentinel's harness-engineering pass: closing the reliability gap that let a single malformed tool call crash a turn and dump a raw exception to the user, and getting off a Groq model that is scheduled for shutdown.
Companion reading: `docs/learning/05-harness-engineering.html` (the research this spec acts on).
Last updated: 2026-07-11.

**Revision note.** The first version of this spec was reviewed by Codex (`gpt-5.6-sol`, medium effort) against the real installed `langchain==1.3.11` source before implementation started.
That review found two blocking design errors, which this revision corrects: `ToolRetryMiddleware` cannot retry tool-body exceptions the way `create_agent` wires `ToolNode` (verified directly, see "Tool execution failure" below - it's dropped from this spec, not fixed), and the `HumanInTheLoopMiddleware` adapter's assumption that the frontend could keep resolving confirmations one at a time was wrong (`Command(resume=...)` requires every bundled decision at once - the frontend design changes instead, see below).
It also found scope this spec had missed entirely: the CLI and the snapshot endpoint both need their own migration, not just the SSE surface.

## Goal (and non-goals)

Every category in the harness field notes' failure map gets a LangChain-native middleware, except context overflow and tool execution failure (see "Tool execution failure" - dropped after review, not deferred by choice).
The two Groq models Sentinel is pinned to (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) are both on Groq's own deprecation list and get swapped.

Explicit non-goals for this slice:

- **No context-window management.**
  `SummarizationMiddleware` / `ContextEditingMiddleware` are deferred - nothing in the failure map's likelihood column for this category is urgent yet, and it is a genuinely separate concern from reliability.
- **No observability/evals build-out.**
  LangSmith tracing and an eval suite are their own future spec.
  This slice only touches failure *handling*, not failure *measurement*.
- **No PII middleware.**
  Single-user learning backend; not carrying real user data.
- **No "edit" decision on confirmations.**
  `HumanInTheLoopMiddleware` supports approve/edit/reject; Sentinel keeps today's binary approve/reject scope.
  Editing a proposed tool call's arguments before approving it is a real future feature, not part of this pass.

## Why now

Two independent triggers landed in the same session:

1. **A live incident.**
   Mid-conversation, the model answered in prose and also called `save_config()` with no arguments, before the conversation had gathered any of its four required fields.
   Groq validates tool-call arguments against the JSON schema server-side; the call was rejected with an HTTP 400 `tool_use_failed` before any normal completion came back.
   Nothing in `sentinel/api/app.py` or `sentinel/agents/tools.py` catches a model-call-level exception - it propagated straight through `create_agent`'s loop and out the SSE endpoint as a raw exception dump.
2. **A dying model.**
   Groq's deprecation page lists both `llama-3.3-70b-versatile` (Sentinel's `"smart"` tier) and `llama-3.1-8b-instant` (its `"cheap"` tier) with a scheduled shutdown around 2026-08-16 on free/developer tiers.
   Confirmed directly against `console.groq.com/docs/deprecations` before treating this as settled, since the initial finding came from a scraped summary.

## Model swap

`sentinel/llm/provider.py`'s `_MODELS["groq"]` dict changes from `{"smart": "llama-3.3-70b-versatile", "cheap": "llama-3.1-8b-instant"}` to `{"smart": "openai/gpt-oss-120b", "cheap": "openai/gpt-oss-20b"}` - both are Groq's own recommended replacements for the models being retired.

`gpt-oss-120b` has native tool-calling support, benchmarks at or above OpenAI's own o4-mini on several tasks, and Groq's free tier lists 30 RPM / 1K RPD / 8K TPM / 200K TPD for it - roughly in line with what `llama-3.3-70b-versatile` offers today.
The 8K TPM figure is worth watching once this ships: Sentinel's system prompt embeds the full domain glossary, so a single turn's prompt is not trivial in size.
If TPM turns out to be the binding constraint in practice, `ModelFallbackMiddleware` (below) is the safety valve, not a reason to block this swap.
None of the deprecation dates, rate limits, or benchmark claims above are verifiable from the repo or from installed source - they are Groq's own external documentation, unlike everything else in this spec, which is checked against real code.

This is a self-contained config change with no middleware dependency.
It should land as its own early, low-risk task in the implementation plan, verified against the real API (a live smoke test, not just offline tests) before anything else in this spec is built on top of it.

## Tool execution failure: dropped, not fixed

The original design here (`ToolRetryMiddleware`, scoped to the five non-guarded tools) does not work, and there is no supported way to make it work through `create_agent`'s public interface.

Verified directly against the installed `langgraph` source: `create_agent` constructs its internal `ToolNode` without passing `handle_tool_errors`, so it keeps `ToolNode`'s default - a bare `except Exception` in `_execute_tool_sync` that catches *any* exception raised inside a tool body and converts it directly into an error `ToolMessage`, before `ToolRetryMiddleware.wrap_tool_call`'s own `try/except` ever sees anything to retry.
`create_agent(...)`'s signature does not expose `handle_tool_errors` at all, so there is no clean way to disable this and let exceptions reach the retry middleware instead.

Given that, `ToolRetryMiddleware` is dropped from this spec rather than worked around.
Two things make this an acceptable cut, not a gap:

- Sentinel's tools already catch their own expected failures and return strings (`tools.py`'s own documented convention) - the retry middleware was already flagged in the first draft of this spec as "belt-and-suspenders," not a primary fix.
- `ToolNode`'s default behavior already does the important part for free: any exception that *does* escape a tool body becomes a graceful `ToolMessage` the model can react to, not a crash. The only thing lost by dropping the middleware is automatic retry attempts on tool failures - a nice-to-have with no incident behind it, unlike the model-call failure this spec's other middleware directly fixes.

## The middleware stack

Applied to `create_agent(...)` in `sentinel/agents/agent.py`.
Ordering matters - before-hooks run first-to-last in list order, wrap-hooks nest with the first middleware outermost, after-hooks run in reverse.

```python
settings = get_settings()
middleware=[
    ModelCallLimitMiddleware(                                     # checked before any model call runs at all
        thread_limit=settings.sentinel_model_call_thread_limit,
        run_limit=settings.sentinel_model_call_run_limit,
    ),
    ToolCallLimitMiddleware(
        thread_limit=settings.sentinel_tool_call_thread_limit,
        run_limit=settings.sentinel_tool_call_run_limit,
    ),
    ModelFailureFormatterMiddleware(corrective_feedback),  # outermost of the model-call error-handling layers below - the last resort
    ModelFallbackMiddleware(get_chat_model("smart", name="anthropic")),  # provider outage / rate-limit escalation
    ModelRetryMiddleware(
        max_retries=settings.sentinel_retry_max_attempts,
        on_failure="error",              # re-raise, don't swallow - so fallback (outer) actually gets a turn
    ),
    InvalidToolCallMiddleware(corrective_feedback),    # custom - closes the GH #33504 gap, see below
    HumanInTheLoopMiddleware(                          # innermost: replaces confirm()/interrupt(), see below
        interrupt_on={
            name: {"allowed_decisions": ["approve", "reject"], "when": guarded_when(name)}
            for name in ("train", "retrain", "promote", "delete", "run_monitor")
        },
    ),
]
```

| Failure category (from the field notes) | Middleware | Notes |
|---|---|---|
| Model call failure | `ModelRetryMiddleware` + `ModelFallbackMiddleware` + `ModelFailureFormatterMiddleware` | Direct fix for the `save_config` incident |
| Invalid tool-call parsing | custom `InvalidToolCallMiddleware` | No built-in exists (LangChain GH #33504) |
| Tool execution failure | *(dropped)* | Not achievable through `create_agent`'s public API - see above |
| Runaway loops / cost | `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` | Cheap insurance, thread- and run-scoped |
| Safety / consent | `HumanInTheLoopMiddleware` | Replaces hand-rolled `confirm()` / `interrupt()` |
| Context overflow | *(deferred)* | Out of scope per this spec's non-goals |
| Observability | *(deferred)* | Out of scope per this spec's non-goals |
| Model staleness | model swap, above | Config change, not a middleware |

## Corrective feedback: deterministic first, cheap-tier LLM as fallback

`corrective_feedback(error: Exception) -> str` is one shared function, used by two different middleware in two different ways - worth being precise about both, since a first pass at this design got the layering wrong (caught while translating this spec into the implementation plan, corrected here rather than silently in the plan).

**Where it plugs in.**

- `InvalidToolCallMiddleware(corrective_feedback)` - the return value becomes a `ToolMessage`, which the agent reads on its very next model call *within the same turn*. This is the "diagnose and act, without ending the conversation" path.
- `ModelFailureFormatterMiddleware(corrective_feedback)` - a small new outermost middleware (not a built-in) that only runs once retry *and* fallback have both been exhausted for a model-call failure. Its return value becomes an `AIMessage` - the turn ends gracefully, and the message is what the user sees directly, in place of a raw exception dump. This is the direct fix for the `save_config` incident: instead of a crash, the user sees something like *"I still need a few details before I can save your configuration: failure_threshold, reporting_cadence, success_metric."*

Both call sites get the same two-tier message-building logic:

1. **Deterministic templates, keyed on exception shape.**
   Groq's own `tool_use_failed` error already names the missing schema fields (`missing properties: 'framing', 'failure_threshold', ...`) - parse that list directly and template `"That call is missing required fields: {fields}. Ask the user for these before calling it again."`
   A `KeyError` (unknown `model_id`) templates to a message naming valid ids, pulled live from `registry.list()`.
   Worth being honest about this branch: Sentinel's own tools already catch `KeyError` themselves and return a string rather than raising, so this template is not exercised by any current tool call - it stays in as cheap, correct, defensive coverage for a shape that could reach `corrective_feedback` from elsewhere (or in the future), not as evidence it currently fires.
   This tier is instant, free, deterministic, and testable with plain assertions.
2. **Cheap-tier fallback, for the unclassified residual.**
   Only reached when tier 1's pattern matching doesn't recognize the exception.
   A single, stateless call to `tools_chat_model` (the existing `"cheap"` tier, already wired for report grounding) with **no chat history** - just the raw error, asked to produce one sentence naming what went wrong and what to try.
   No new model wiring needed; this reuses the tier Sentinel already has.

Note the asymmetry this implies: an invalid-tool-call failure lets the agent keep working the same turn; a model-call failure (nothing was ever generated to act on) can only end the turn gracefully.
Retry and fallback exist precisely to make that second, harder-to-recover case as rare as possible before it happens.

## The `invalid_tool_calls` gap

`create_agent`'s routing only checks whether `AIMessage.tool_calls` is empty before ending a turn; when the model's JSON is malformed enough that LangChain's own parser can't build a `tool_calls` entry, it lands in `invalid_tool_calls` instead and the turn silently ends without the model ever seeing what went wrong (LangChain issue #33504, open and unfixed upstream as of this research).

No built-in middleware covers this.
The Codex review flagged a simpler mechanism than the first draft used: `create_agent`'s routing honors an explicit `jump_to` directive from middleware *before* it even looks at `tool_calls` (`_resolve_jump`, checked first in both `model_to_tools` and `model_to_model`).
So `InvalidToolCallMiddleware` doesn't need to synthesize a fake `tool_calls` entry and a matching `ToolMessage` together (the first draft's approach, which worked but fought the routing logic instead of using its front door) - an `after_model` hook can instead append one `ToolMessage` per invalid call (`tool_call_id` taken from the invalid call, content from `corrective_feedback`) and return `{"messages": [...], "jump_to": "model"}`, sending execution straight back to the model with the correction already in context.

This is still the one piece of the stack with no reference implementation to lean on; write it test-first against a fake chat model that returns a scripted `invalid_tool_calls` payload.

## Replacing `confirm()` / `interrupt()` with `HumanInTheLoopMiddleware`

Agreed after a real back-and-forth in this session: consistency across the whole harness stack is worth the integration cost.
The Codex review found the integration cost was bigger than the first draft accounted for, on the frontend side specifically - corrected here.

**The contract mismatch.** Today, each guarded tool (`train`, `retrain`, `promote`, `delete`, `run_monitor`) calls `interrupt()` itself, mid-body.
Two guarded calls in one turn produce two separate interrupts, each with its own id in `state.interrupts`, resumed via `Command(resume={interrupt_id: answer, ...})` - this id-addressed shape is a deliberate V2 design choice, made specifically to avoid a scalar/positional resume bug.
`HumanInTheLoopMiddleware` works differently: it intercepts *before* a guarded tool runs (in `after_model`, before the tool node runs at all - see the retry-interaction note below), and bundles every simultaneously-pending guarded call into **one** `interrupt()` call whose value is `{"action_requests": [{"name", "args", "description"}, ...], "review_configs": [...]}` (verified against the installed `langchain==1.3.11` source).
Resume is `Command(resume={"decisions": [{"type": "approve"}, {"type": "reject", "message": "..."}, ...]})` - the middleware requires **exactly one decision per bundled action, submitted together in one call** (verified: it raises `ValueError` if the counts don't match). There is no supported way to answer one bundled action now and another later.

**What this means for the frontend, corrected from the first draft.** The first draft assumed the existing UI (each `ConfirmCard` resolves independently and fires its own `/resume` request immediately) could stay as-is behind a server-side id-adapter.
It can't - `HumanInTheLoopMiddleware` has no notion of a partial answer, so the server would have to buffer decisions itself until every bundled action is answered, adding real statefulness for no clear benefit.
Instead, the frontend changes: clicking Yes/No on a `ConfirmCard` now records a local choice without submitting anything; once every currently-pending card has a choice, a single "Submit responses" action sends all of them together in one `/resume` call.
This is a real, visible UX change from today's "each card resolves the instant you click it" - a deliberate trade accepted in this session for a simpler, correct server side rather than a stateful buffering layer.

The API contract itself (`POST /sessions/{id}/resume` with `{"answers": {"<id>": "yes"|"no", ...}}`) does not need to change shape - only its meaning does (a client always sends the full batch at once now, matching what `HumanInTheLoopMiddleware` requires directly, so `_stream()`'s translation is a straightforward one-to-one mapping instead of needing an id-to-position adapter with partial-submission handling).

**Also in scope, missed by the first draft:**

- **The CLI** (`sentinel/agents/__main__.py`) drives confirmations through the same old shape (`state.interrupts[0].value["tool"]`/`["detail"]`, scalar `Command(resume={id: answer})`) and needs its own migration to the new `action_requests`/`decisions` shape - not just the HTTP API.
- **`GET /sessions/{id}`** (the snapshot endpoint) currently spreads each interrupt's value directly into `pending_confirmations`; under `HumanInTheLoopMiddleware` there is one interrupt whose value nests an `action_requests` list, so this endpoint needs the same expansion logic `_stream()` uses, not a pass-through.

**The retry-middleware interaction, corrected.** The first draft justified excluding the five guarded tools from `ToolRetryMiddleware`'s scope as protection against a documented `GraphInterrupt`-vs-retry conflict.
Since `ToolRetryMiddleware` is dropped entirely (see above), this protection is moot - but it's worth recording that the original reasoning was also wrong post-migration anyway: `HumanInTheLoopMiddleware` raises its interrupt from `after_model`, before the tool node runs at all, so no `wrap_tool_call`-based retry middleware would ever have wrapped that interrupt regardless of list order or tool scoping.

## Config surface

New fields on `sentinel/config.py`'s `Settings`, matching the existing 12-factor convention (env + `.env`, no hardcoded constants):

```python
sentinel_model_call_thread_limit: int = 40
sentinel_model_call_run_limit: int = 15
sentinel_tool_call_thread_limit: int = 40
sentinel_tool_call_run_limit: int = 20
sentinel_retry_max_attempts: int = 2
```

Model names stay on the existing `sentinel_model_smart` / `sentinel_model_cheap` override fields - no new fields needed there, just the new defaults in `_MODELS["groq"]`.

## Testing

Offline, same discipline as the rest of the agent layer - no live LLM, no PyCaret.

- **A new fake in `tests/fakes.py`**: a chat model whose `_generate` raises a scripted exception shaped like Groq's real `tool_use_failed` `BadRequestError` (same "missing properties" message format), so the tier-1 corrective-feedback path is tested against the actual incident's shape, not a generic exception.
- **A fake for `invalid_tool_calls`**: a small custom fake that returns an `AIMessage` with empty `tool_calls` and a populated `invalid_tool_calls` on its first call only (later calls behave normally), so a test can prove both that `InvalidToolCallMiddleware` catches what `create_agent`'s default routing misses *and* that the loop actually continues afterward.
- **Per-middleware unit coverage**: call-limit middleware actually ends a run at the configured limit; fallback middleware actually switches models on a scripted failure; retry middleware respects `max_retries` and stops.
- **A genuinely batched confirmation test**: two guarded tool calls inside *one* `AIMessage` (not two sequential turns, the first draft's mistake) produce one interrupt with two `action_requests`, exposed as two independently-addressable API confirmation entries, resolved together by one ordered resume.
- **CLI and snapshot-endpoint coverage** for their own migrations, alongside the SSE surface.
- **A model-swap smoke test**: one live (not offline) check that `openai/gpt-oss-120b` actually completes a tool call through `ChatGroq` + `bind_tools`, run manually before merging, not part of the offline CI suite.

## What this unlocks (deliberately deferred)

- Context management (`SummarizationMiddleware`, `ContextEditingMiddleware`) once conversations are long enough to need it.
- Observability (LangSmith tracing) and an eval suite (DeepEval/Promptfoo), so production failures like the incident that started this spec become regression fixtures instead of one-off debugging sessions.
- The "edit" decision on `HumanInTheLoopMiddleware`, letting a user adjust a proposed tool call's arguments instead of only approve/reject.
- A revisit of tool-execution-failure handling if a real incident ever motivates it - possible paths include a custom `wrap_tool_call` that intercepts before `ToolNode`'s own error handling, or asking upstream LangChain for a way to pass `handle_tool_errors` through `create_agent`.
