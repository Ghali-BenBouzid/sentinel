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
- `pending_prompt() -> str | None` - the interviewer question currently awaiting an answer.
- `answer(text)` - hands the user's reply to the blocked `ask` (puts it on the in-queue).
- `done -> bool` and `final_state() -> AgentState | None` - whether `invoke` returned, and the final
  state (report, alerts, log).
- `error -> Exception | None` - any unexpected exception raised in the thread (see Error handling).

An `Event` is a tiny record `{kind, payload}` where `kind` is one of the strings above. Two
`queue.Queue`s (out for events, in for answers) plus a completion flag are the entire concurrency
surface; that is deliberately the smallest primitive that works.

### `app.py` - the Streamlit view

One page holding a single `GraphRunner` in `st.session_state` across reruns. Four sections gated on
the derived phase (`interview -> training -> report -> monitor -> done`). Phase is *derived* by the
UI from the events it has seen plus whether `final_state()` is set; the graph's internal `event`
field never leaks into the UI.

## Data flow (the bridge)

Two interaction modes, split by phase, because a blocking graph and a rerun-on-interaction UI need
different handling.

### Interview phase = rerun-driven (natural chat)

```
thread: ask("Q1")  -> puts ("prompt","Q1") on out-queue, then blocks on in-queue.get()
UI rerun: poll() drains ("prompt","Q1") -> append to chat, show st.chat_input
user submits -> runner.answer(text) puts on in-queue -> st.rerun()
thread: ask() unblocks -> classify_turn (one LLM call) -> ask("Q2") -> repeat
```

Each user message is one rerun. The bot's reply appears on the next rerun once the thread posts the
next prompt. Between submit and next prompt there is a ~1-2s LLM call, so after `answer()` the UI
shows a "thinking" spinner and runs a short bounded poll-loop until the next `prompt` (or a phase
change) arrives.

### Training + monitor phases = in-script polling loop (no rerun churn)

When `poll()` returns `training_started`, the script enters a live block that updates placeholders
in place (Streamlit permits updating placeholders inside a running script):

```python
with st.status("Comparing model families...") as status:
    while not runner.saw("training_finished"):
        for ev in runner.poll():
            ...  # append notifications to the status, update elapsed timer
        time.sleep(0.5)
# then render leaderboard + best model + held-out RMSE/MAE/R2 from final_state
```

Monitor is the same shape but fast: once `final_state()` is available, the UI animates through
`final["alerts"]` as a stepper (these are the *real* decisions, revealed progressively - the monitor
computes them in one batched pass) and renders a card per filed ticket.

**Design decision:** the training/monitor live view uses an in-script `while` + `time.sleep(0.5)`
poll loop rather than a third-party auto-refresh component. This avoids an extra dependency and is
the standard Streamlit idiom; one script run is "parked" polling during training, which is fine for
a single-user demo. Approved over the auto-refresh-component alternative.

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

- `tests/test_dashboard_runner.py` - one offline unit test for `GraphRunner`, using the *same*
  fakes `tests/test_agents.py` already uses (fake providers, stub `train_fn`) plus scripted answers
  fed through `answer()`. Asserts the queue bridge carries a full conversation to a completed
  `final_state`: prompts come out, answers go in, report + alerts land. No Streamlit, no live LLM,
  no PyCaret.
- `app.py` is a thin view and is not unit-tested (testing Streamlit scripts needs its own harness;
  YAGNI for an MVP). It is exercised by running it.

## Packaging and running

- Add `streamlit` to a new `dashboard` optional-dependency group in `pyproject.toml`, not the core
  deps - `uv sync` for the graph/tests stays lean; the demo installs the extra.
- README gains a short "Run the dashboard" section:

  ```bash
  uv sync --extra dashboard
  uv run streamlit run sentinel/dashboard/app.py
  ```
