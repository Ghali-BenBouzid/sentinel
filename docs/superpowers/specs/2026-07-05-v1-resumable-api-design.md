# V1 - resumable agent layer + streaming HTTP API (design)

This is the design for Sentinel V1.
It supersedes the throwaway Streamlit demo with a framework-native, resumable graph exposed as a streaming HTTP API that a real front end (built in a separate, parallel session) consumes.
Last updated: 2026-07-05.

## Goal (and non-goals)

V1 delivers a **backend API only**.
We own and define the HTTP contract; no UI is built in this session.

The one deliverable, stated in one line:
make the agent layer resumable statelessly across requests (LangGraph `interrupt()` + a checkpointer), and expose it as a FastAPI + Server-Sent-Events API the front end drives.

Explicit non-goals for V1:

- **No `create_agent` adoption.**
  It is a prebuilt ReAct tool-loop; none of the current sub-agents are that shape.
  The interviewer is a scripted agenda (code owns order and bounds), the report writer is a single grounded `complete()` call, the monitor is a threshold check.
  `create_agent` earns its place in V2, when the monitor becomes a genuinely autonomous, tool-using agent taking real actions.
- **No real front end here.**
  It is being built in a parallel session and will consume this API.
- **No auth / multi-tenant model.**
  Single-user learning backend.
  `thread_id` is the session identifier; add auth when there is a reason to.

## Why the refactor (not framework-chasing)

The existing `GraphRunner` runs the graph on a daemon thread and bridges the interviewer's blocking `ask()`/`notify()` seam to the UI through queues.
That works for one Streamlit process.
It does not survive a real web backend: a blocking-thread-per-session model cannot span independent HTTP requests, cannot survive a process restart, and cannot scale horizontally.

`interrupt()` + a checkpointer is the framework-native answer: the graph runs until it needs human input, `interrupt()`s (persisting its state via the checkpointer), returns control to the caller, and resumes on the next request via `Command(resume=...)`.
No thread, no queues, no injected `ask`/`notify`.
This is the *only* justification for touching the load-bearing interviewer loop, and it is the sanctioned V1 exception to the "do not touch the interviewer loop" rule in `AGENTS.md`.

## The three seams we replace

| Today | V1 | Why |
|---|---|---|
| injected `ask(q) -> str` (blocks a thread) | `interrupt(q)` from `langgraph.types` | graph suspends and persists, resumes on the next request; no thread |
| injected `notify(msg)` | `get_stream_writer()` custom stream events | progress and applied-defaults announcements stream out natively, no injected callable |
| `GraphRunner` (daemon thread + queues) + Streamlit | `SqliteSaver` checkpointer + `graph.stream(...)` over FastAPI/SSE | resumable statelessly across HTTP requests; survives restart |

**Untouched:** `train_fn`, `provider_smart`, `provider_cheap`, and `ticket_dir` still inject via `config["configurable"]`.
That seam is what keeps the whole graph testable offline with fakes, and it is orthogonal to human input.
Only `ask` and `notify` leave the `configurable` dict.

## Component 1 - the self-looping interviewer node

`interrupt()` is **not** a drop-in for `ask()`.
When a LangGraph node resumes after an `interrupt()`, it re-executes from the top of the node function; prior `interrupt()`s return their saved values instantly, but every other side effect re-runs.
The current interviewer makes one `classify_turn` LLM call *per turn* inside a single node, so a naive `ask` -> `interrupt` swap would re-run every earlier turn's LLM call on each resume (extra cost, and non-deterministic on the weak free-tier model).

The fix is to make each turn its own checkpointed graph step.
One node invocation = exactly one question/answer round:

```python
def interviewer_turn(state):
    prompt = state["next_prompt"]     # computed at the end of the prior turn; cheap, from the checkpoint
    reply = interrupt(prompt)          # the ONLY interrupt; the node suspends here
    return advance(state, reply)       # exactly ONE classify_turn LLM call; updates state + next_prompt, or marks the interview done
```

Routing: a conditional edge sends `done -> orchestrator`, else `-> interviewer_turn`.
On resume the node re-runs from the top, but `interrupt` returns the saved reply immediately and `advance` runs its single LLM call once.
There is **no replay of earlier turns**, because each earlier turn was its own already-checkpointed invocation.

### What moves, what stays identical

The LLM-facing content does not change.
`_SYSTEM_PROMPT`, `_GATE_SYSTEM_PROMPT`, the five classifications (CLEAR / UNCLEAR / QUESTION / WANTS_DEFAULT / ALL_DEFAULTS), the deduction gate (`DEDUCE_CONFIDENCE`), the `MAX_NONANSWERS` bound, the `DEFAULTS`, and the all-defaults fast path all carry over verbatim.

What changes is control flow: the blocking `while True` loop in `_resolve_field` and the `for field in QUESTIONS` loop in `run_interview` become pure functions over explicit interview state.
The loop-local variables (`values`, `deduced`, `resolved`, `history`, `preamble`, `nonanswers`, the active field index, and the gate/short-circuit phase) become fields the graph checkpoints between turns.

`advance(state, reply)` is the state machine:

1. classify `reply` for the active field (one `classify_turn` call), or run `classify_gate` on the opening gate turn.
2. absorb confident deductions for still-open fields.
3. apply the outcome (CLEAR -> store value and move on, WANTS_DEFAULT / MAX_NONANSWERS -> default, ALL_DEFAULTS or gate-accept -> fill remaining defaults and finish, QUESTION / UNCLEAR -> re-ask same field).
4. compute `next_prompt` (the next field's question, a deduced-value confirmation, or a re-ask) with the ack from this turn prepended, or set `done` and write the final `InterviewConfig` into state.

Applied-default announcements and the closing acknowledgement that used to go through `notify` become `get_stream_writer()` events (see Component 3).

This is a real rewrite of `interviewer.py`'s control flow, but it keeps "code owns the agenda, LLM owns the language" - arguably making it a more explicit state machine than the loop it replaces.

### Interview state

`AgentState` gains a nested interview-progress field (only meaningful while the interview runs), or an `InterviewProgress` TypedDict is threaded alongside `config`.
Fields: `active_index`, `values`, `deduced`, `resolved`, `history`, `next_prompt`, `phase` (`gate` | `field` | `done`), `nonanswers`.
`config` (the finished `InterviewConfig`) is still written once, on `done`, exactly as today, so downstream nodes are unchanged.

## Component 2 - the graph and the checkpointer

- `build_graph()` compiles with `checkpointer=SqliteSaver(...)`.
  SqliteSaver is one file, no external service, and survives a process restart - which is the whole point.
  New dependency: `langgraph-checkpoint-sqlite`.
- The interviewer becomes the self-looping `interviewer_turn` node with a conditional self-edge; every other node (orchestrator, trainer, report_writer, monitor) is unchanged.
- Every graph invocation now requires `config["configurable"]["thread_id"]` (the session id) so the checkpointer can persist and resume per session.
- The checkpoint DB path is configurable via `sentinel/config.py` (a new setting, defaulting to a local file under `artifacts/`).

## Component 3 - notify becomes a custom stream

Inside any node, `get_stream_writer()` returns a writer; `writer({"type": "notify", "text": ...})` emits a custom event that `graph.stream(..., stream_mode="custom")` surfaces to the caller.
This replaces every current `notify(...)` call (applied-default announcements, the interviewer's closing line, and any progress the trainer wants to surface).
No injected callable, and it works identically under the CLI driver and the HTTP driver.

Trainer progress: the trainer node brackets the long PyCaret run with `writer({"type": "training", "phase": "started"|"finished"})` events, replacing the `GraphRunner` wrapping that did this before.

## Component 4 - the FastAPI surface

Server-to-client streaming uses Server-Sent Events; client-to-server uses plain POST.
SSE is simpler than WebSocket and sufficient, because the interaction is strictly turn-based (the server streams until it needs input, the client POSTs one answer).

Endpoints:

- `POST /sessions`
  Creates a new `thread_id`, starts the graph, and returns an SSE stream of events (`notify`, `training`, `report`, and finally a `prompt` event) up to the first `interrupt`, then holds or closes.
  Response includes the `thread_id`.
- `POST /sessions/{thread_id}/resume` with body `{"answer": "..."}`
  Resumes the graph with `Command(resume=answer)` and returns an SSE stream up to the next `interrupt` or `END`.
- `GET /sessions/{thread_id}`
  Returns the full transcript / current state reconstructed from the checkpointer, for reconnect and page reload.
  The checkpointer is the single source of truth; nothing lives only in a browser session.

Event shapes streamed over SSE (JSON per event): `prompt` (a question awaiting an answer), `notify` (a status/applied-default line), `training` (started/finished), `report` (the final report text), `done` (terminal, with a short summary), `error`.

The FastAPI app translates one `graph.stream(input, config, stream_mode=["custom","updates"])` iteration into that SSE event sequence.
It reads providers, `train_fn`, and `ticket_dir` from settings/DI exactly as `__main__.py` does today, and injects them into `config["configurable"]` per request.

New dependencies: `fastapi` and `uvicorn` as **core** deps (the API is the V1 product, and its tests are the primary verification, so they must run in CI - not hidden behind an optional extra the way the dashboard was).
The `dashboard` optional-extra is removed.

## Component 5 - the CLI driver

`python -m sentinel.agents` stays as the local end-to-end driver, but its body changes from a single `graph.invoke` to a small stream/resume loop over the same interrupt API:

```
for event in graph.stream({"event": "start"}, config):
    if <interrupt>:  reply = scripted_or_input();  resume with Command(resume=reply)
    else:            print the streamed event
```

`--interactive` uses `input()`; the default uses the existing `SCRIPTED_ANSWERS`.
This keeps a zero-dependency, offline way to run the whole flow, and it exercises the same interrupt/stream path the API uses.

## What gets deleted

- `sentinel/dashboard/` (`runner.py`, `app.py`, `__init__.py`).
- `tests/test_dashboard_app.py`, `tests/test_dashboard_runner.py`.
- The `dashboard` optional-dependency in `pyproject.toml` and its mentions.

Rationale: the Streamlit dashboard was explicitly throwaway; its job was to de-risk the graph in a browser, and the API plus its tests replace it as the "prove it works" surface.
Keeping it would mean maintaining two interview drivers (the blocking `ask`/`notify` path and the interrupt path) - dead flexibility.
`docs/HANDOFF.md`, `AGENTS.md`/`CLAUDE.md`, and `README.md` get updated to describe the API instead of the dashboard.

## Testing and verification

- **Unit / graph-offline (existing style):** the whole graph still runs offline with a fake provider and fake `train_fn` injected via `config["configurable"]`, now driven through `graph.stream` + `Command(resume=...)` with a `thread_id` and an in-memory or temp-file checkpointer.
  Existing `tests/test_agents.py` assertions on the interview outcomes are preserved (the classifications and defaults did not change), adapted to the stream/resume driver.
- **Interviewer replay safety (new, the critical check):** a test that drives a multi-turn interview and asserts the fake provider is called exactly once per turn - i.e. resuming turn N does not re-invoke `classify_turn` for turns 1..N-1.
  This is the one behaviour the whole refactor exists to get right, so it gets an explicit regression test.
- **HTTP end-to-end (new):** FastAPI `TestClient` + a fake provider + fake `train_fn` drives `POST /sessions` -> `resume` -> ... -> `done` fully offline, asserting the SSE event sequence and that `GET /sessions/{id}` reconstructs the transcript after a simulated reconnect.
  Browser automation is dead in the WSL dev environment, but this is an HTTP API, so `TestClient` is the correct and complete verification surface - the browser limitation does not bite.
- CI runs these because FastAPI and the sqlite checkpointer are core deps, not optional extras.

## Lessons to carry forward (do not regress)

- CV leaderboard vs held-out test are different measurements; selection stays on cross-validation (never rank/select by the test set).
- Keep numeric logic out of the weak model; the success verdict stays in `report_writer._success_verdict`, the report writer stays grounding-constrained.
- Domain knowledge stays in `sentinel/agents/domain_context.py`; the LLM prompts stay in code, not scattered.
- LLM access stays behind the `sentinel/llm/provider.py` seam.

## Open questions for the implementation plan

- Exact SSE framing (one long-lived stream per request vs close-after-interrupt) and how the front end reconnects to an in-flight training run.
  Default: close the stream at each interrupt; the front end reopens on resume, and `GET /sessions/{id}` covers reload.
- Whether `InterviewProgress` is a nested field on `AgentState` or a sibling threaded through the interviewer node only.
  Lean: a dedicated field on `AgentState`, cleared once `config` is written.
