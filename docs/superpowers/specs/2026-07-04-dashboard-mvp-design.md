# Sentinel dashboard MVP - design

The deferred "dashboard (later)" layer from `docs/pdm-agent-design.md`, built to make the
project demo-worthy.
This is a throwaway Streamlit prototype: it turns the existing M2 graph (interview -> train ->
report -> monitor) into something a person can watch in a browser, without changing the graph.
A real web frontend comes later and will reuse the same backend seam this spec introduces.

## Goal and scope

- **Goal:** a single-page web dashboard that runs the *real* M2 graph end to end, live, and shows
  all four phases: the interview chat, training, the report, and the monitor + tickets.
- **Training is live.** No pre-baked artifact shortcut - the demo runs PyCaret for real (it caches
  FD001 on disk as the M1 pipeline already does, and caches the running `GraphRunner` in the
  Streamlit session so a stray rerun does not retrain).
- **The M2 graph is frozen.** Nothing in `sentinel/agents/` or `sentinel/llm/` changes. The
  dashboard drives the graph purely through the existing `config["configurable"]` injection seam.
- **Out of scope:** authentication, deployment/hosting, multi-user, persistence beyond the local
  tickets the monitor already writes, streaming PyCaret's internal per-family progress, and any
  rework of the interviewer loop. These are explicitly not in this MVP.

## Architecture

A new fourth layer that sits *beside* the graph, never inside it:

```
DS core (M1) -> agent layer (M2, frozen) -> dashboard (this)
```

Two new files under a new `sentinel/dashboard/` package, plus one optional dependency:

```
sentinel/
  dashboard/
    runner.py    # GraphRunner: runs graph.invoke in a thread, bridges ask/notify via queues. NO streamlit import.
    app.py       # Streamlit UI: 4 sections, reads GraphRunner via st.session_state. ALL st.* here.
```

The split is load-bearing: `runner.py` is framework-agnostic and is the seam the *real* web app
will reuse later; `app.py` is a disposable view. The background thread only ever touches queues,
never `st.*` (background threads have no Streamlit ScriptRunContext - touching `st.*` from one is
the classic Streamlit-threading bug this design avoids by construction).

## Components

### `GraphRunner` (runner.py) - the only real new logic

A plain, framework-agnostic class. Public API:

- `start()` - spawns a daemon thread running
  `graph.invoke({"event": "start"}, config={"configurable": {...}})`, injecting:
  - a queue-backed `ask(question) -> str` (puts a `prompt` event on the out-queue, blocks on the
    in-queue for the reply, returns it),
  - a queue-backed `notify(msg)` (puts a `notify` event on the out-queue, non-blocking),
  - the real `train_fn` wrapped to emit `training_started` / `training_finished` markers around
    `sentinel.agents.training.run_training`,
  - `provider_smart`, `provider_cheap` from `get_provider(...)`, and `ticket_dir`.
- `poll() -> list[Event]` - drains and returns every event emitted since the last call
  (`prompt`, `notify`, `training_started`, `training_finished`, `error`). Non-blocking, thread-safe.
- `saw(kind) -> bool` - convenience: whether an event of that kind has been emitted at any point
  (the runner remembers kinds it has emitted, so the UI polling loop can test `saw("training_finished")`
  without having to have caught that exact event in its own `poll()` call).
- `history() -> list[Event]` - the full append-only transcript (prompts, answers, notifications,
  training markers) the UI renders from.
- `pending_prompt() -> str | None` - the interviewer question currently awaiting an answer.
- `answer(text)` - clears the pending prompt synchronously, records the answer in the transcript,
  then unblocks the worker's `ask`.
- `training_elapsed() -> float | None` - seconds since training started (for the live timer).
- `done -> bool` and `final_state() -> AgentState | None` - whether `invoke` returned, and the final
  state (report, alerts, log).
- `error -> Exception | None` - any unexpected exception raised in the thread (see Error handling).

An `Event` is a tiny record `{kind, payload}` where `kind` is one of the strings above. The
concurrency surface is one append-only `history` list (worker appends events; the UI thread appends
answers and reads) plus one `queue.Queue` the worker blocks on for the next answer - deliberately the
smallest primitive that works.

### `app.py` - the Streamlit view

One page that renders as a pure function of the `GraphRunner`'s transcript. The runner is held in
`st.cache_resource` (not `st.session_state`) so it survives reruns *and* fresh sessions (refreshes).
Four sections gated on the derived phase (`interview -> training -> report -> monitor -> done`).
Phase is *derived* by the UI from the events it has seen plus whether `final_state()` is set; the
graph's internal `event` field never leaks into the UI.

## Data flow (the bridge)

> **Revision (2026-07-04, after live testing).** The first cut split the UI into a rerun-driven
> chat plus an in-script `while`-loop for training, coordinated through `st.session_state`. Three
> bugs came out of that: a browser refresh restarted from the Start screen, training finished but
> the report/monitor never rendered (a dead `st.stop()` with nothing to wake it), and answering a
> question re-rendered the same question because `pending_prompt()` was read stale right after
> `answer()`. Root cause: the view was **not a pure function of runner state** and it raced the
> worker on a mutable flag. The section below describes the shipped design that replaced it. The
> app.py docstring is the living source of truth.

The view is a **pure function of an append-only transcript** the runner exposes as
`history()` (bot prompts, applied-default notifications, and the user's own answers, in order),
plus a few state reads (`pending_prompt()`, `saw()`, `done`, `final_state()`). Every rerun rebuilds
the whole page from that transcript, so a fresh page load (browser refresh) reproduces the exact
same page.

Two rules make the concurrency safe:

- `answer(text)` clears the pending prompt **synchronously** (on the UI thread) and records the
  answer in the transcript, *then* unblocks the worker. So the run immediately after an answer never
  sees the just-answered prompt as still pending, and the next reply can never land on the wrong
  field.
- The worker thread only appends to `history` and reads the answer queue; it never touches `st.*`.

```
Start click -> build GraphRunner, start() its thread, store it in st.cache_resource
thread: ask("Q1") -> history += prompt("Q1"), set pending="Q1", block on answer
UI run: render history; pending? -> show st.chat_input, wait (worker is blocked)
user submits -> runner.answer(text): pending=None + history += answer(text) + unblock -> st.rerun()
UI run: render history; pending is None, not done -> show "Thinking..." -> sleep(0.5) + st.rerun()
thread: ask() unblocks -> classify_turn (one LLM call) -> ask("Q2") -> history += prompt("Q2")
UI run: pending="Q2" -> show st.chat_input ... (repeat until interview_done -> training)
```

**One rerun point drives all progress.** While the run is active and *not* waiting on the user
(`not done` and no pending prompt), the script does `time.sleep(0.5); st.rerun()`. That single poll
covers both the "thinking" gap between interview turns and the multi-minute training wait (the
training section shows an elapsed timer via `runner.training_elapsed()`). When a prompt is pending
the script shows the chat box and stops (the worker is blocked, so there is nothing to poll). When
`done`, it renders the leaderboard + metrics + report + monitor alerts/tickets and settles with no
further rerun. Monitor alerts are the *real* decisions the graph already computed (batched), read
from `final_state()["alerts"]`.

**Refresh-survival:** the `GraphRunner` is held in `st.cache_resource` (a process-global singleton),
not `st.session_state`. A browser refresh is a fresh session but reconnects to the same running
runner and rebuilds the page from its transcript; the background thread never stopped.
(ponytail: a global singleton is a single-user-demo simplification, not multi-user safe; a real web
app would key the runner per session/user.)

**Design decision:** a `sleep + st.rerun()` busy-poll rather than a third-party auto-refresh
component - no extra dependency, and fine for one viewer. A real web app would use a push channel
(websocket/SSE) instead.

## Error handling

- **Training failure is already modelled by the graph.** On a PyCaret exception the trainer sets
  `event="run_failed"`, the orchestrator still routes to the report writer (so the failure gets
  explained), and it ends without the monitor. The runner surfaces that final state unchanged; the
  UI renders the failure report and skips the monitor section - mirroring the graph's own fork, not
  reimplementing it.
- **Unexpected thread exceptions** (anything not handled inside the graph) are caught by the
  runner, stored on `error`, and emitted as an `error` event, so the UI shows an error box instead
  of hanging forever on a prompt that will never arrive.
- **Missing key / provider misconfig:** the config layer (`sentinel/config.py`) already raises; the
  runner catches it and the UI shows a clear message rather than a terminal stack trace.

## Testing

- `tests/test_dashboard_runner.py` - offline unit tests for `GraphRunner`, using the *same* fakes
  `tests/test_agents.py` uses (fake providers, stub `train_fn`) plus scripted answers fed through
  `answer()`. Asserts the bridge carries a full conversation to a completed `final_state` (prompts
  out, answers in, report + alerts land), that `answer()` clears the pending prompt synchronously,
  and that `history()` is an append-only transcript including the user's answers. No Streamlit, no
  live LLM, no PyCaret.
- `tests/test_dashboard_app.py` - runs the real `app.py` via Streamlit's `AppTest` (LLM + training
  faked) and locks the three UI-bug fixes: the run self-advances after each answer, the report +
  monitor render once training finishes, and a fresh session (refresh) reconnects instead of
  restarting. Guarded by `pytest.importorskip("streamlit")`, so CI without the `dashboard` extra
  stays green.

## Packaging and running

- Add `streamlit` to a new `dashboard` optional-dependency group in `pyproject.toml`, not the core
  deps - `uv sync` for the graph/tests stays lean; the demo installs the extra.
- README gains a short "Run the dashboard" section:

  ```bash
  uv sync --extra dashboard
  uv run streamlit run sentinel/dashboard/app.py
  ```
