# Sentinel Dashboard MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a demo-worthy single-page Streamlit dashboard that runs the *real, unchanged* M2 LangGraph graph end to end (interview -> train -> report -> monitor) and shows all four phases live in the browser.

**Architecture:** A new `sentinel/dashboard/` package sits *beside* the frozen agent layer. `runner.py` holds a framework-agnostic `GraphRunner` that runs `graph.invoke` in a daemon thread and bridges the graph's blocking `ask`/`notify` seam to the UI through thread-safe queues. `app.py` is a throwaway Streamlit view that only reads from `GraphRunner`. The background thread never touches `st.*`; the UI never touches threads.

**Tech Stack:** Python 3.11, LangGraph (existing), Streamlit (new optional extra), stdlib `threading` + `queue`.

## Global Constraints

- Python `>=3.11,<3.12` (PyCaret constraint - already pinned in `pyproject.toml`).
- **The M2 agent layer is frozen:** do NOT modify anything under `sentinel/agents/` or `sentinel/llm/`. The dashboard drives the graph only through the existing `config["configurable"]` seam (`ask`, `notify`, `provider_smart`, `provider_cheap`, `train_fn`, `ticket_dir`).
- **`runner.py` must not import `streamlit`** and must never call `st.*` - it is the reusable backend seam for the future real web app.
- **The background thread must never touch `st.*`** (background threads have no Streamlit ScriptRunContext) - it communicates only through the runner's queues.
- Training is live (no pre-baked artifact); PyCaret runs for real via the existing `sentinel.agents.training.run_training`.
- Streamlit is an *optional* dependency (`[project.optional-dependencies].dashboard`), never a core dependency - `uv sync` and CI stay lean.
- No em dashes in prose/comments (use `-`). Match the repo's existing docstring + comment style.
- Lint clean: `uv run ruff check .` (pyflakes + pycodestyle + isort import order).

---

## File Structure

- Create: `sentinel/dashboard/__init__.py` - package marker (empty).
- Create: `sentinel/dashboard/runner.py` - `GraphRunner` + `Event`. The only real new logic. Framework-agnostic.
- Create: `sentinel/dashboard/app.py` - Streamlit view. All `st.*` lives here. Not unit-tested.
- Create: `tests/test_dashboard_runner.py` - offline unit tests for `GraphRunner`, reusing the fakes from `tests/test_agents.py`.
- Modify: `pyproject.toml` - add the `dashboard` optional-dependency extra.
- Modify: `README.md` - add a "Run the dashboard" section.

---

## Task 1: `GraphRunner` - the threaded queue bridge

**Files:**
- Create: `sentinel/dashboard/__init__.py`
- Create: `sentinel/dashboard/runner.py`
- Test: `tests/test_dashboard_runner.py`

**Interfaces:**
- Consumes: `sentinel.agents.graph.build_graph()` (compiled graph; `.invoke(state, config=...)`), the `config["configurable"]` seam keys, `sentinel.agents.state.AgentState`.
- Produces (later tasks / the app rely on these exact names):
  - `Event` dataclass: `Event(kind: str, payload: object = None)`. `kind` in `{"prompt", "notify", "training_started", "training_finished", "error"}`.
  - `GraphRunner(*, graph, provider_smart, provider_cheap, train_fn, ticket_dir: str)`.
  - `GraphRunner.start() -> None` - spawn the daemon thread (idempotent: a second call is a no-op).
  - `GraphRunner.poll() -> list[Event]` - drain and return events emitted since the last call (non-blocking, thread-safe).
  - `GraphRunner.pending_prompt() -> str | None` - the interview question currently awaiting an answer, else `None`.
  - `GraphRunner.answer(text: str) -> None` - deliver the user's reply to the blocked `ask` (no-op if nothing is pending).
  - `GraphRunner.saw(kind: str) -> bool` - whether an event of that kind has ever been emitted.
  - `GraphRunner.done -> bool` (property) - whether `graph.invoke` has returned or raised.
  - `GraphRunner.error -> Exception | None` (property) - an unexpected exception raised by the thread, else `None`.
  - `GraphRunner.final_state() -> AgentState | None` - the final graph state once `done`, else `None`.

- [ ] **Step 1: Create the package marker**

Create `sentinel/dashboard/__init__.py`:

```python
"""Dashboard layer: a Streamlit view over the M2 agent graph (see runner.py)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_dashboard_runner.py`. It reuses the agent-layer fakes to drive a full conversation through the bridge with no live LLM and no PyCaret.

```python
"""Offline tests for the dashboard's GraphRunner (the threaded queue bridge).

No live LLM, no PyCaret, no Streamlit: the graph is real but its dependencies are
the same fakes tests/test_agents.py uses. We drive the interview by polling for
prompts and calling answer(), exactly as the Streamlit view does, and assert the
bridge carries a full run to a completed final_state.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from sentinel.agents.graph import build_graph
from sentinel.agents.state import InterviewConfig
from sentinel.agents.training import TrainingRun
from sentinel.core.automl import TrainResult
from sentinel.dashboard.runner import Event, GraphRunner


class QueueProvider:
    """A Provider that returns queued replies in order (last one repeats)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, **kwargs):
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        return self.replies[i]


class FakeProvider:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, **kwargs):
        return self.reply


def _turn(classification, reply, value=None, deduced=None):
    return json.dumps(
        {"classification": classification, "reply": reply, "value": value, "deduced": deduced or []}
    )


def _gate(all_defaults):
    return json.dumps({"all_defaults": all_defaults})


def _fake_run():
    lb = pd.DataFrame({"Model": ["Extra Trees"], "MAE": [11.9], "RMSE": [17.1], "R2": [0.82]})
    result = TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 11.9, "r2": 0.82},
        model_path="artifacts/rul_model.pkl",
        metrics_path="artifacts/metrics.json",
    )
    test_eval = pd.DataFrame({"unit": [1, 2], "RUL": [10, 90]})
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: list(f["RUL"]))


def _wait(pred, timeout=5.0):
    """Spin until pred() is true or timeout (keeps threaded tests from hanging)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_runner_bridges_full_interview_to_completed_state(tmp_path):
    # Gate declined, then one CLEAR turn per field (4 fields) -> 5 asks total.
    smart = QueueProvider(
        [
            _gate(False),
            _turn("CLEAR", "ok", "turbofan RUL"),
            _turn("CLEAR", "ok", 30),
            _turn("CLEAR", "ok", "each run"),
            _turn("CLEAR", "ok", "RMSE < 20"),
        ]
    )
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("Report: the model is good."),
        train_fn=lambda cfg: _fake_run(),
        ticket_dir=str(tmp_path),
    )
    runner.start()

    answers = iter(["no, let's tune it", "turbofan RUL", "30", "each run", "RMSE < 20"])
    # Answer every prompt the interviewer emits until the graph completes.
    while True:
        assert _wait(lambda: runner.pending_prompt() is not None or runner.done), "runner stalled"
        if runner.done:
            break
        runner.answer(next(answers, ""))

    assert _wait(lambda: runner.done)
    assert runner.error is None
    final = runner.final_state()
    assert isinstance(final["config"], InterviewConfig)
    assert final["config"].failure_threshold == 30
    assert final["report"] == "Report: the model is good."
    assert final["event"] == "monitor_done"
    assert [a["unit"] for a in final["alerts"]] == [1]
    assert (tmp_path / "ticket_unit_1.json").exists()
    # The training lifecycle markers were emitted for the UI to react to.
    assert runner.saw("training_started") and runner.saw("training_finished")


def test_runner_surfaces_training_failure_without_hanging(tmp_path):
    def boom(cfg):
        raise RuntimeError("PyCaret exploded")

    smart = QueueProvider([_gate(True)])  # take the all-defaults fast path, skip Q&A
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("unused"),
        train_fn=boom,
        ticket_dir=str(tmp_path),
    )
    runner.start()

    # Gate is the only prompt; accept defaults, then the graph trains (and fails).
    assert _wait(lambda: runner.pending_prompt() is not None)
    runner.answer("yes, just use defaults")

    assert _wait(lambda: runner.done), "runner hung on training failure"
    final = runner.final_state()
    # The graph itself reports the failure and stops before the monitor.
    assert final["event"] == "failed_reported"
    assert "PyCaret exploded" in final["error"]
    assert "alerts" not in final
    assert runner.error is None  # a graph-handled failure is NOT an unexpected thread error


def test_poll_returns_notify_events(tmp_path):
    # All-defaults path announces each applied default via notify -> Event(kind="notify").
    smart = QueueProvider([_gate(True)])
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("Report."),
        train_fn=lambda cfg: _fake_run(),
        ticket_dir=str(tmp_path),
    )
    runner.start()
    assert _wait(lambda: runner.pending_prompt() is not None)
    runner.answer("yes, use defaults")
    assert _wait(lambda: runner.done)

    kinds = set()
    # Drain everything the run emitted (poll is incremental).
    for _ in range(50):
        for ev in runner.poll():
            assert isinstance(ev, Event)
            kinds.add(ev.kind)
        if runner.done and not runner.pending_prompt():
            break
        time.sleep(0.01)
    assert "notify" in kinds
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_dashboard_runner.py -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinel.dashboard.runner'` (or `ImportError` for `Event`/`GraphRunner`).

- [ ] **Step 4: Implement `GraphRunner`**

Create `sentinel/dashboard/runner.py`:

```python
"""GraphRunner: run the M2 graph in a background thread, bridged to a UI by queues.

The M2 interviewer talks to a human through a *blocking* ``ask(question) -> str``
callable, and the graph runs start-to-finish in one ``graph.invoke`` call. A
rerun-on-interaction UI (Streamlit) cannot call a blocking function on its own
thread. So we run ``graph.invoke`` on a daemon thread and bridge the graph's
injected ``ask``/``notify`` seam to the UI through two thread-safe queues:

- ``ask(q)``  -> push an Event(kind="prompt", payload=q) onto the OUT queue,
                 then block on the IN queue until the UI calls ``answer(text)``.
- ``notify(m)`` -> push an Event(kind="notify", payload=m) onto the OUT queue.

The training function is wrapped to bracket the (long, live) PyCaret run with
``training_started`` / ``training_finished`` events so the UI can show progress.

This module is deliberately framework-agnostic: it imports no ``streamlit`` and
the worker thread never touches ``st.*``. It is the seam a future real web app
reuses; only ``app.py`` is throwaway.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from ..agents.state import AgentState


@dataclass
class Event:
    """One thing the running graph emitted for the UI to react to."""

    kind: str  # "prompt" | "notify" | "training_started" | "training_finished" | "error"
    payload: object = None


class GraphRunner:
    """Drive a compiled graph on a background thread, bridged to a UI by queues."""

    def __init__(self, *, graph, provider_smart, provider_cheap, train_fn, ticket_dir: str) -> None:
        self._graph = graph
        self._provider_smart = provider_smart
        self._provider_cheap = provider_cheap
        self._train_fn = train_fn
        self._ticket_dir = ticket_dir

        self._out: "queue.Queue[Event]" = queue.Queue()  # graph -> UI
        self._in: "queue.Queue[str]" = queue.Queue()  # UI -> graph (interview answers)
        self._seen_kinds: set[str] = set()
        self._pending_prompt: str | None = None
        self._final: AgentState | None = None
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread once (a second call is a no-op)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        configurable = {
            "ask": self._ask,
            "notify": self._notify,
            "provider_smart": self._provider_smart,
            "provider_cheap": self._provider_cheap,
            "train_fn": self._wrapped_train_fn,
            "ticket_dir": self._ticket_dir,
        }
        try:
            self._final = self._graph.invoke({"event": "start"}, config={"configurable": configurable})
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
            self._error = exc
            self._emit("error", f"{type(exc).__name__}: {exc}")

    # --- seam callables injected into the graph (run on the worker thread) ---

    def _ask(self, question: str) -> str:
        """Blocking ask: publish the prompt, then wait for the UI's answer."""
        self._pending_prompt = question
        self._emit("prompt", question)
        answer = self._in.get()  # blocks the worker until answer() is called
        self._pending_prompt = None
        return answer

    def _notify(self, message: str) -> None:
        self._emit("notify", message)

    def _wrapped_train_fn(self, config):
        """Bracket the real training run with lifecycle events for the UI."""
        self._emit("training_started")
        run = self._train_fn(config)  # may raise; the trainer node catches it (run_failed)
        self._emit("training_finished")
        return run

    def _emit(self, kind: str, payload: object = None) -> None:
        self._seen_kinds.add(kind)
        self._out.put(Event(kind, payload))

    # --- UI-facing API (run on the main / Streamlit thread) ---------------

    def poll(self) -> list[Event]:
        """Drain and return every event emitted since the last poll."""
        events: list[Event] = []
        while True:
            try:
                events.append(self._out.get_nowait())
            except queue.Empty:
                break
        return events

    def pending_prompt(self) -> str | None:
        return self._pending_prompt

    def answer(self, text: str) -> None:
        """Hand the user's reply to the blocked ask (no-op if nothing pending)."""
        if self._pending_prompt is not None:
            self._in.put(text)

    def saw(self, kind: str) -> bool:
        return kind in self._seen_kinds

    @property
    def done(self) -> bool:
        return self._final is not None or self._error is not None

    @property
    def error(self) -> Exception | None:
        return self._error

    def final_state(self) -> AgentState | None:
        return self._final
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_dashboard_runner.py -v`
Expected: PASS - all three tests green.

- [ ] **Step 6: Run the full suite + lint (no regressions in the frozen layer)**

Run: `uv run pytest && uv run ruff check .`
Expected: all tests pass, ruff reports no issues.

- [ ] **Step 7: Commit**

```bash
git add sentinel/dashboard/__init__.py sentinel/dashboard/runner.py tests/test_dashboard_runner.py
git commit -m "feat(dashboard): GraphRunner bridges the M2 graph to a UI via queues"
```

---

## Task 2: Streamlit view, packaging, and docs

**Files:**
- Create: `sentinel/dashboard/app.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies].dashboard`)
- Modify: `README.md` (add "Run the dashboard" section)

**Interfaces:**
- Consumes: `GraphRunner`, `Event` from Task 1; `sentinel.config.get_settings`, `sentinel.llm.provider.get_provider`, `sentinel.agents.graph.build_graph`, `sentinel.agents.training.run_training`.
- Produces: a runnable Streamlit app (`streamlit run sentinel/dashboard/app.py`). No programmatic consumers.

- [ ] **Step 1: Add the optional dependency extra**

In `pyproject.toml`, after the `[build-system]` block (before `[dependency-groups]`), add:

```toml
[project.optional-dependencies]
# The demo dashboard only - kept out of the core deps so `uv sync` and CI stay lean.
dashboard = ["streamlit>=1.40"]
```

- [ ] **Step 2: Sync the extra**

Run: `uv sync --extra dashboard`
Expected: resolves and installs `streamlit` into `.venv`; `uv.lock` updates.

- [ ] **Step 3: Write the Streamlit view**

Create `sentinel/dashboard/app.py`. It holds one `GraphRunner` in `st.session_state`, renders four phase-gated sections, and never touches threads directly.

```python
"""Streamlit dashboard: watch the M2 agent graph run end to end, live.

This is a throwaway demo view. All Streamlit calls live here; the graph itself is
driven by GraphRunner (sentinel/dashboard/runner.py), which is framework-agnostic.

Run it:

    uv sync --extra dashboard
    uv run streamlit run sentinel/dashboard/app.py

Two interaction modes, split by phase (see the dashboard design doc):
- Interview  -> rerun-driven chat (one rerun per user message).
- Train/monitor -> an in-script poll loop that updates placeholders in place,
  so the long live training run does not fight Streamlit's rerun model.
"""

from __future__ import annotations

import time

import streamlit as st

from ..agents.graph import build_graph
from ..agents.training import run_training
from ..config import get_settings
from ..llm.provider import get_provider
from .runner import GraphRunner

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="centered")


def _drain_into_chat(runner: GraphRunner) -> None:
    """Move any new prompt/notify events into the persistent chat transcript."""
    for ev in runner.poll():
        if ev.kind == "prompt":
            st.session_state.chat.append(("assistant", ev.payload))
        elif ev.kind == "notify":
            st.session_state.chat.append(("assistant", f"_{ev.payload}_"))
        elif ev.kind == "error":
            st.session_state.chat.append(("assistant", f"**Error:** {ev.payload}"))


def _start_run() -> None:
    """Construct providers + GraphRunner and start the worker thread."""
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=get_provider("smart"),
        provider_cheap=get_provider("cheap"),
        train_fn=run_training,
        ticket_dir="artifacts/tickets",
    )
    runner.start()
    st.session_state.runner = runner
    st.session_state.chat = []


st.title("🛰️ Sentinel")
st.caption("Predictive-maintenance agent - interview → train → report → monitor, live.")

# --- start screen ---------------------------------------------------------
if "runner" not in st.session_state:
    provider = get_settings().sentinel_llm_provider
    st.write(
        "This runs the real agent graph against NASA C-MAPSS FD001. "
        f"LLM provider: **{provider}**. Training runs PyCaret live, so it takes a few minutes."
    )
    if st.button("Start", type="primary"):
        try:
            _start_run()
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - show config/key errors in the UI, not the console
            st.error(f"Could not start: {type(exc).__name__}: {exc}")
    st.stop()

runner: GraphRunner = st.session_state.runner
_drain_into_chat(runner)

# --- 1. interview chat ----------------------------------------------------
st.subheader("1 · Interview")
for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

if runner.error is not None:
    st.error(f"The run failed unexpectedly: {runner.error}")
    st.stop()

if runner.pending_prompt() is not None:
    reply = st.chat_input("Your answer")
    if reply:
        st.session_state.chat.append(("user", reply))
        runner.answer(reply)
        st.rerun()
    st.stop()  # wait for the user; nothing further to render this run

# If the interview is still resolving (LLM thinking) and training hasn't begun,
# spin briefly for the next prompt so the chat feels responsive.
if not runner.saw("training_started") and not runner.done:
    with st.spinner("Thinking..."):
        if _wait_for(runner, lambda: runner.pending_prompt() is not None or runner.saw("training_started") or runner.done):
            st.rerun()

# --- 2. training ----------------------------------------------------------
if runner.saw("training_started"):
    st.subheader("2 · Training")
    if not runner.saw("training_finished") and not runner.done:
        started = time.time()
        with st.status("Comparing model families with PyCaret...", expanded=True) as status:
            while not runner.saw("training_finished") and not runner.done:
                for ev in runner.poll():
                    if ev.kind == "notify":
                        st.write(ev.payload)
                status.update(label=f"Training... ({int(time.time() - started)}s elapsed)")
                time.sleep(0.5)
            status.update(label="Training complete", state="complete")
        st.rerun()

# From here on we need the finished graph state.
if not runner.done:
    st.stop()

final = runner.final_state()

# A training failure is reported by the graph itself; render it and stop.
if final.get("event") == "failed_reported":
    st.subheader("2 · Training")
    st.error("Training did not complete.")
    st.subheader("3 · Report")
    st.write(final.get("report", "(no report)"))
    st.stop()

# Successful run: show the leaderboard + headline metrics.
run = final["train_run"]
st.dataframe(run.result.leaderboard, use_container_width=True)
m = run.result.metrics
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{m['rmse']:.1f}")
c2.metric("MAE", f"{m['mae']:.1f}")
c3.metric("R²", f"{m['r2']:.2f}")

# --- 3. report ------------------------------------------------------------
st.subheader("3 · Report")
st.write(final.get("report", "(no report)"))

# --- 4. monitor + tickets -------------------------------------------------
st.subheader("4 · Monitor")
alerts = final.get("alerts", [])
n_alerts = sum(a["decision"] == "alert" for a in alerts)
st.write(f"{len(alerts)} readings flagged · {n_alerts} maintenance tickets filed.")
for a in alerts:
    icon = "🔴" if a["decision"] == "alert" else "🟠"
    line = f"{icon} unit {a['unit']}: predicted RUL {a['predicted_rul']} → {a['decision']}"
    if a.get("ticket"):
        line += f"  · ticket: `{a['ticket']}`"
    st.write(line)
```

- [ ] **Step 4: Add the `_wait_for` helper used by the view**

`app.py` references `_wait_for`; define it near the top of `app.py`, right after the `_drain_into_chat` function:

```python
def _wait_for(runner: GraphRunner, pred, timeout: float = 30.0) -> bool:
    """Spin (draining events) until pred() is true or timeout. UI-thread only."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        _drain_into_chat(runner)
        if pred():
            return True
        time.sleep(0.2)
    return False
```

- [ ] **Step 5: Byte-compile the app to catch syntax/name errors**

Run: `uv run python -m py_compile sentinel/dashboard/app.py`
Expected: no output (exit 0). This catches typos and undefined names without needing a live LLM key.

- [ ] **Step 6: Smoke-boot the app headless (no key needed to reach the start screen)**

Run:
```bash
uv run streamlit run sentinel/dashboard/app.py --server.headless true --server.port 8599 &
sleep 8
curl -sf http://localhost:8599/ >/dev/null && echo "APP BOOTED OK"
kill %1
```
Expected: prints `APP BOOTED OK` (the start screen renders without constructing any provider, so no API key is required to verify boot).

- [ ] **Step 7: Add the README section**

In `README.md`, immediately after the "Run the agent graph end to end" section (before "## Tests, lint, and CI"), add:

```markdown
### Run the demo dashboard

A Streamlit dashboard runs the whole agent graph in the browser - chat through the
interview, watch PyCaret train live, read the report, and step through the monitor's
alerts and filed tickets. Streamlit is an optional extra (kept out of the core deps):

\```bash
uv sync --extra dashboard
uv run streamlit run sentinel/dashboard/app.py
\```

It uses the same provider config as the CLI (`SENTINEL_LLM_PROVIDER` + your key from
`.env`), runs the real graph end to end, and writes the same mock tickets to
`artifacts/tickets/`. Training runs PyCaret live, so the training step takes a few
minutes - the dashboard shows a live status while it compares model families.
```

(Remove the backslashes before the code fences - they are escaped here only so this plan renders.)

- [ ] **Step 8: Lint and full test suite**

Run: `uv run ruff check . && uv run pytest`
Expected: ruff clean, all tests pass (the app is not tested; Task 1's runner tests still pass).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock README.md sentinel/dashboard/app.py
git commit -m "feat(dashboard): Streamlit view + packaging + docs for the live demo"
```

---

## Self-Review

**1. Spec coverage:**
- Goal (single-page dashboard, real graph, four phases live) -> Task 2 `app.py` (four `st.subheader` sections). ✓
- Live training, session-cached runner -> Task 2 (`run_training` as `train_fn`; runner held in `st.session_state`). ✓
- M2 frozen, driven only via `configurable` -> Task 1 `_run` builds the `configurable` dict; no `sentinel/agents` edits in any task. ✓
- `sentinel/dashboard/` sibling package, `runner.py` (no streamlit) + `app.py` (all st.*) -> Tasks 1 and 2; Global Constraints restate the import ban. ✓
- `GraphRunner` API (`start/poll/pending_prompt/answer/saw/done/error/final_state`) -> Task 1 implements every method. ✓
- Data flow: interview = rerun-driven; train/monitor = in-script poll loop -> Task 2 `app.py` (chat_input + `st.rerun` vs. the `st.status` while-loop). ✓
- Error handling: graph-reported training failure rendered + monitor skipped; unexpected thread exception surfaced; missing key shown in UI -> Task 1 (`error` capture, `failed_reported` passthrough) + Task 2 (start-screen try/except, `failed_reported` branch, `runner.error` box). ✓
- Testing: one offline `GraphRunner` test reusing existing fakes; app not unit-tested -> Task 1 Step 2 (three tests) + Task 2 (py_compile + headless smoke boot instead). ✓
- Packaging: streamlit as optional extra; README section -> Task 2 Steps 1, 7. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" - every step has concrete code or an exact command. ✓

**3. Type consistency:** `Event(kind, payload)`, `GraphRunner(*, graph, provider_smart, provider_cheap, train_fn, ticket_dir)`, and every method name are identical between Task 1's implementation, its tests, and Task 2's `app.py` usage. `final_state()` returns the graph's `AgentState` dict, and `app.py` reads exactly the keys the graph sets (`event`, `train_run`, `report`, `alerts`, `error`) as confirmed against `state.py`/`monitor.py`/`graph.py`. ✓
```
