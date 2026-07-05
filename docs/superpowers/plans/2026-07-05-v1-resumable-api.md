# V1 Resumable Agent Layer + Streaming HTTP API - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking `ask()`/`notify()` + `GraphRunner` thread bridge with a framework-native, resumable graph (LangGraph `interrupt()` + `SqliteSaver`) exposed as a FastAPI/SSE API.

**Architecture:** The interviewer becomes a self-looping graph node where one invocation is exactly one question/answer round (a single `interrupt()`, a single LLM call), so resume never replays earlier turns. `notify` becomes `get_stream_writer()` custom events. The graph compiles with a checkpointer keyed by `thread_id`; a thin FastAPI layer drives it with `graph.stream` + `Command(resume=...)` and forwards events as SSE. The `python -m sentinel.agents` CLI stays as a local stream/resume driver.

**Tech Stack:** Python 3.11, LangGraph 1.2.7 (`interrupt`, `Command`, `get_stream_writer`, `SqliteSaver`), FastAPI + uvicorn, pytest, uv.

## Global Constraints

- Python `>=3.11,<3.12` (PyCaret pin). Do not change.
- LLM access only through `sentinel/llm/provider.py` (the `Provider` protocol). Never import `anthropic`/`groq` elsewhere.
- Domain facts only from `sentinel/agents/domain_context.py`. Do not inline domain knowledge into prompts.
- The interviewer's LLM prompts, classifications (CLEAR / UNCLEAR / QUESTION / WANTS_DEFAULT / ALL_DEFAULTS), deduction gate (`DEDUCE_CONFIDENCE=0.6`), `MAX_NONANSWERS=3`, and `DEFAULTS` carry over **verbatim** - only control flow changes.
- Dependencies that are not data (`provider_smart`, `provider_cheap`, `train_fn`, `ticket_dir`) inject via `config["configurable"]`. Only `ask`/`notify` are removed from it.
- Config is read via `sentinel/config.py` `get_settings()` (pydantic-settings, `lru_cache`d). Tests that mutate config env must call `get_settings.cache_clear()`.
- No AI co-author trailers or "Generated with..." footers on commits.
- Each full sentence on its own line in Markdown docs.
- `report_writer._success_verdict` stays in code; the report writer stays grounding-constrained. Do not touch.

---

## File Structure

- `pyproject.toml` - add `fastapi`, `uvicorn`, `langgraph-checkpoint-sqlite` to core deps; add `httpx` to dev; remove the `dashboard` optional extra.
- `sentinel/config.py` - add `checkpoint_db_path` setting.
- `sentinel/agents/state.py` - add `InterviewProgress` and its field on `AgentState`.
- `sentinel/agents/interviewer.py` - rewrite control flow into a pure `advance()` state machine + a self-looping `interviewer_turn` node. Prompts/classification logic unchanged.
- `sentinel/agents/graph.py` - self-loop edge for the interviewer; compile with a checkpointer; trainer emits stream-writer progress.
- `sentinel/agents/__main__.py` - CLI becomes a stream/resume loop.
- `sentinel/api/app.py` (new) - FastAPI app: `POST /sessions`, `POST /sessions/{id}/resume`, `GET /sessions/{id}`, SSE encoding.
- `sentinel/api/__init__.py` (new).
- `tests/test_interviewer_state.py` (new) - pure state-machine + replay-safety tests.
- `tests/test_agents.py` - adapt to the stream/resume driver.
- `tests/test_api.py` (new) - FastAPI `TestClient` end-to-end.
- Delete: `sentinel/dashboard/`, `tests/test_dashboard_app.py`, `tests/test_dashboard_runner.py`.
- Docs: `docs/HANDOFF.md`, `CLAUDE.md`/`AGENTS.md`, `README.md`.

---

### Task 1: Dependencies + checkpoint config setting

**Files:**
- Modify: `pyproject.toml`
- Modify: `sentinel/config.py`
- Test: `tests/test_config.py` (add one test; create if absent)

**Interfaces:**
- Produces: `get_settings().checkpoint_db_path` (str, default `"artifacts/sentinel-checkpoints.sqlite"`).

- [ ] **Step 1: Edit `pyproject.toml` dependencies**

In `[project].dependencies` add:
```toml
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "langgraph-checkpoint-sqlite>=2.0",
```
Delete the entire `[project.optional-dependencies]` block (the `dashboard` extra).
In `[dependency-groups].dev` add `"httpx>=0.28"` (FastAPI `TestClient` needs it).

- [ ] **Step 2: Add the setting to `sentinel/config.py`**

Add a field to the settings class (match the existing field style):
```python
    checkpoint_db_path: str = "artifacts/sentinel-checkpoints.sqlite"
```

- [ ] **Step 3: Write the failing test**

In `tests/test_config.py`:
```python
def test_checkpoint_db_path_has_default():
    from sentinel.config import get_settings
    get_settings.cache_clear()
    assert get_settings().checkpoint_db_path.endswith(".sqlite")
```

- [ ] **Step 4: Sync and run**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: PASS, and `uv run python -c "from langgraph.checkpoint.sqlite import SqliteSaver; print('ok')"` prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock sentinel/config.py tests/test_config.py
git commit -m "build(v1): add fastapi/uvicorn/sqlite-checkpointer deps + checkpoint path setting"
```

---

### Task 2: Interview state model + pure `advance()` state machine

This is the core logic move. The blocking loop in `interviewer.py` becomes a pure function `advance(progress, reply, provider)` that consumes one user reply, makes at most one classifier call, and returns the next `InterviewProgress`. No graph, no `interrupt` yet - fully unit-testable.

**Files:**
- Modify: `sentinel/agents/state.py`
- Modify: `sentinel/agents/interviewer.py`
- Test: `tests/test_interviewer_state.py` (create)

**Interfaces:**
- Produces `InterviewProgress` (TypedDict, `total=False`) with fields:
  `phase: str` (`"gate"|"field"|"done"`), `active_index: int`, `values: dict`, `deduced: dict`, `resolved: list[str]`, `history: list[str]`, `next_prompt: str`, `nonanswers: int`, `notices: list[str]` (applied-default/ack lines to stream), `config: InterviewConfig | None`.
- Produces `start_progress() -> InterviewProgress` (phase `"gate"`, `next_prompt=GATE_QUESTION`, empty collections).
- Produces `advance(progress: InterviewProgress, reply: str, provider: Provider) -> InterviewProgress` - pure apart from exactly one provider call per turn; on the gate turn it calls `classify_gate`, otherwise `classify_turn`.
- Consumes existing `classify_gate`, `classify_turn`, `_absorb_deductions`, `_coerce`, `DEFAULTS`, `_EXTRA_DEFAULTS`, `_DEFAULT_LINE`, `_CONFIRM`, `QUESTIONS`, `_ASKED_FIELDS`, `MAX_NONANSWERS`, `GATE_QUESTION` (all already in `interviewer.py`, unchanged).

- [ ] **Step 1: Add `InterviewProgress` to `sentinel/agents/state.py`**

```python
class InterviewProgress(TypedDict, total=False):
    """Per-turn interview state the graph checkpoints between interviewer turns."""
    phase: str            # "gate" | "field" | "done"
    active_index: int     # index into interviewer.QUESTIONS
    values: dict          # resolved field -> value
    deduced: dict         # field -> confidently-deduced value awaiting confirmation
    resolved: list        # fields already resolved (order-independent)
    history: list         # "Assistant: ..."/"User: ..." lines for LLM context
    next_prompt: str      # the message to interrupt() with on the next turn
    nonanswers: int       # consecutive non-answers on the active field
    notices: list         # applied-default / ack lines to stream this turn
    config: object        # the finished InterviewConfig once phase == "done"
```
Add `interview: InterviewProgress` to `AgentState` (import `InterviewProgress` at top of the class body / module).

- [ ] **Step 2: Write failing tests for the state machine**

In `tests/test_interviewer_state.py`:
```python
from sentinel.agents import interviewer as iv
from sentinel.agents.state import InterviewConfig


class OneShotProvider:
    """Fake provider returning a scripted JSON string per complete() call."""
    def __init__(self, replies): self._it = iter(replies); self.calls = 0
    def complete(self, messages, **kw):
        self.calls += 1
        return next(self._it)


def test_gate_accept_fills_all_defaults_in_one_turn():
    p = OneShotProvider(['{"all_defaults": true}'])
    prog = iv.advance(iv.start_progress(), "yes just use defaults", p)
    assert prog["phase"] == "done"
    assert isinstance(prog["config"], InterviewConfig)
    assert prog["config"].failure_threshold == iv.DEFAULTS["failure_threshold"]
    assert p.calls == 1  # exactly one classifier call this turn
    assert any("default" in n.lower() for n in prog["notices"])


def test_gate_decline_then_clear_answer_advances_one_field():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no let's go through it", OneShotProvider(['{"all_defaults": false}']))
    assert prog["phase"] == "field"
    assert prog["active_index"] == 0
    clear = '{"classification":"CLEAR","reply":"Got it.","value":"turbofan RUL","deduced":[]}'
    prog = iv.advance(prog, "predict turbofan RUL", OneShotProvider([clear]))
    assert prog["values"]["framing"] == "turbofan RUL"
    assert prog["active_index"] == 1  # moved to failure_threshold


def test_all_defaults_midway_fills_the_rest():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", OneShotProvider(['{"all_defaults": false}']))
    ad = '{"classification":"ALL_DEFAULTS","reply":"ok defaults","value":null,"deduced":[]}'
    prog = iv.advance(prog, "just use defaults for the rest", OneShotProvider([ad]))
    assert prog["phase"] == "done"
    assert prog["config"].success_metric == iv.DEFAULTS["success_metric"]
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_interviewer_state.py -v`
Expected: FAIL (`advance`/`start_progress` not defined).

- [ ] **Step 4: Implement `start_progress` and `advance` in `interviewer.py`**

Keep every existing function (`classify_gate`, `classify_turn`, `_absorb_deductions`, `_coerce`, `_fill_defaults_and_finish` logic, prompts) as-is. Add the pure state machine that reuses them:

```python
def start_progress() -> "InterviewProgress":
    return {
        "phase": "gate", "active_index": 0, "values": {}, "deduced": {},
        "resolved": [], "history": [], "next_prompt": GATE_QUESTION,
        "nonanswers": 0, "notices": [], "config": None,
    }


def _finish(values: dict, notices: list, opener: str) -> "InterviewProgress":
    """Fill every unanswered asked field with its default, recording announcements."""
    notices.append(opener)
    for f in _ASKED_FIELDS:
        if f not in values:
            notices.append(f"  - {_DEFAULT_LINE[f]}")
            values[f] = DEFAULTS[f]
    return {"phase": "done", "notices": notices,
            "config": InterviewConfig(**values, **_EXTRA_DEFAULTS)}


def _prompt_for(field: str, question: str, deduced: dict, preamble: str) -> str:
    body = _CONFIRM[field].format(v=deduced[field]) if deduced.get(field) is not None else question
    return f"{preamble}\n\n{body}".strip() if preamble else body


def advance(progress, reply, provider):
    """Consume one user reply; return the next InterviewProgress. One LLM call/turn."""
    p = {**progress, "notices": []}  # notices are per-turn

    if p["phase"] == "gate":
        if classify_gate(reply, provider):
            return _finish(p["values"], p["notices"],
                           "Great - using sensible defaults across the board:")
        p["phase"] = "field"
        field, question = QUESTIONS[0]
        p["next_prompt"] = _prompt_for(field, question, p["deduced"], "")
        return p

    field, question = QUESTIONS[p["active_index"]]
    ded_value = p["deduced"].get(field)
    p["history"] = [*p["history"], f"Assistant: {p['next_prompt']}", f"User: {reply}"]
    turn = classify_turn(field, question, p["history"], reply, ded_value, provider)
    _absorb_deductions(p["deduced"], turn.deduced, active=field, resolved=set(p["resolved"]))

    if turn.classification == ALL_DEFAULTS:
        return _finish(p["values"], p["notices"], "Okay - filling everything else with defaults:")

    ack = ""
    if turn.classification == CLEAR:
        value = _coerce(field, turn.value)
        if value is None and ded_value is not None:
            value = _coerce(field, ded_value)
        if value is not None:
            p["values"][field] = value; ack = turn.reply
        else:
            turn.classification = UNCLEAR
    if turn.classification == WANTS_DEFAULT:
        p["values"][field] = DEFAULTS[field]
        ack = turn.reply or f"Okay, I'll go with the default: {DEFAULTS[field]}."
    if turn.classification == QUESTION:
        p["next_prompt"] = turn.reply or question       # re-ask same field, no advance
        return p
    if turn.classification == UNCLEAR:
        p["nonanswers"] += 1
        if p["nonanswers"] < MAX_NONANSWERS:
            p["next_prompt"] = turn.reply or f"I need a clearer answer. {question}"
            return p
        p["values"][field] = DEFAULTS[field]
        ack = f"Let's not get stuck - I'll use the default of {DEFAULTS[field]}."

    # field resolved -> advance
    p["resolved"] = [*p["resolved"], field]
    p["nonanswers"] = 0
    nxt = p["active_index"] + 1
    if nxt >= len(QUESTIONS):
        if ack: p["notices"] = [*p["notices"], ack]
        p["phase"] = "done"
        p["config"] = InterviewConfig(**p["values"], **_EXTRA_DEFAULTS)
        return p
    p["active_index"] = nxt
    nf, nq = QUESTIONS[nxt]
    p["next_prompt"] = _prompt_for(nf, nq, p["deduced"], ack)
    return p
```
Add `from .state import InterviewProgress` where the other `state` imports are. Leave `run_interview` in place for now (Task 5 removes it once nothing calls it).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_interviewer_state.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add sentinel/agents/state.py sentinel/agents/interviewer.py tests/test_interviewer_state.py
git commit -m "feat(interviewer): pure advance() state machine over InterviewProgress"
```

---

### Task 3: Self-looping `interviewer_turn` node + checkpointed graph

Wire `advance()` into a node that does one `interrupt()` per invocation and loops back to itself, then compile the graph with a checkpointer. This is where the replay-safety property is verified against the real graph.

**Files:**
- Modify: `sentinel/agents/interviewer.py` (add `interviewer_turn` node)
- Modify: `sentinel/agents/graph.py` (self-loop edge + checkpointer)
- Test: `tests/test_interviewer_state.py` (add graph-driven replay test)

**Interfaces:**
- Consumes: `start_progress`, `advance` (Task 2); `interrupt`, `Command` from `langgraph.types`; `SqliteSaver`/`MemorySaver`.
- Produces: `interviewer_turn(state, config) -> dict` (graph node). On `done` it returns `{"config": <InterviewConfig>, "event": "interview_done", "interview": <progress>, ...}`; otherwise `{"interview": <progress>}` and loops.
- Produces: `route_interview(state) -> str` returning `"interviewer_turn"` while `phase != "done"`, else `"orchestrator"`.
- Produces: `build_graph(checkpointer=None)` - defaults to a `SqliteSaver` at `get_settings().checkpoint_db_path`; tests pass a `MemorySaver`.

- [ ] **Step 1: Write the failing replay-safety + flow test**

In `tests/test_interviewer_state.py`:
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


def _drive(graph, cfg, answers):
    """Start the graph, feed each interrupt one scripted answer, return final state."""
    thread = {"configurable": {**cfg, "thread_id": "t1"}}
    result = graph.invoke({"event": "start"}, thread)
    for a in answers:
        if "__interrupt__" not in graph.get_state(thread).next and not graph.get_state(thread).tasks:
            break
        result = graph.invoke(Command(resume=a), thread)
    return result, graph.get_state(thread)


def test_graph_interview_one_llm_call_per_turn():
    from sentinel.agents.graph import build_graph
    # gate declines, then one CLEAR answer per field.
    scripted = ['{"all_defaults": false}'] + [
        '{"classification":"CLEAR","reply":"ok","value":"x","deduced":[]}',
        '{"classification":"CLEAR","reply":"ok","value":25,"deduced":[]}',
        '{"classification":"CLEAR","reply":"ok","value":"daily","deduced":[]}',
        '{"classification":"CLEAR","reply":"ok","value":"rmse<20","deduced":[]}',
    ]
    p = OneShotProvider(scripted)
    cfg = {"provider_smart": p, "provider_cheap": p,
           "train_fn": lambda c: (_ for _ in ()).throw(AssertionError("no train")),
           "ticket_dir": "artifacts/tickets"}
    graph = build_graph(checkpointer=MemorySaver())
    # Only run the interview leg: stop once event == interview_done.
    thread = {"configurable": {**cfg, "thread_id": "t1"}}
    graph.invoke({"event": "start"}, thread)
    for a in ["no", "x", "25", "daily", "rmse<20"]:
        st = graph.get_state(thread)
        if not st.tasks:
            break
        graph.invoke(Command(resume=a), thread)
    # One classifier call per delivered answer (gate + 4 fields) - NO replay.
    assert p.calls == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_interviewer_state.py::test_graph_interview_one_llm_call_per_turn -v`
Expected: FAIL (node not wired / `build_graph` has no `checkpointer` param).

- [ ] **Step 3: Add the node to `interviewer.py`**

```python
from langgraph.types import interrupt


def interviewer_turn(state, config):
    """One question/answer round. Single interrupt(); single classifier call on resume."""
    provider = config["configurable"]["provider_smart"]
    progress = state.get("interview") or start_progress()
    reply = interrupt(progress["next_prompt"])     # suspends; returns the saved reply on resume
    progress = advance(progress, reply, provider)

    writer = _get_writer()
    for line in progress.get("notices", []):
        writer({"type": "notify", "text": line})

    out = {"interview": progress, "log": append_log(state, "interviewer: turn")}
    if progress["phase"] == "done":
        out["config"] = progress["config"]
        out["event"] = "interview_done"
    return out
```
Add a tiny helper so tests without a stream context don't crash:
```python
def _get_writer():
    from langgraph.config import get_stream_writer
    try:
        return get_stream_writer()
    except Exception:      # no active stream (e.g. graph.invoke in tests)
        return lambda _e: None
```
Delete the old `interviewer_node`, `run_interview`, `_resolve_field`, and `_fill_defaults_and_finish` (their behaviour now lives in `advance`/`_finish`/`interviewer_turn`).

- [ ] **Step 4: Rewire and checkpoint the graph in `graph.py`**

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from .interviewer import interviewer_turn


def route_interview(state):
    return "orchestrator" if (state.get("interview") or {}).get("phase") == "done" else "interviewer_turn"


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("interviewer_turn", interviewer_turn)
    graph.add_node("trainer", trainer_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("monitor", monitor_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", route)   # unchanged hub routing
    graph.add_conditional_edges("interviewer_turn", route_interview)
    for node in ("trainer", "report_writer", "monitor"):
        graph.add_edge(node, "orchestrator")

    if checkpointer is None:
        from ..config import get_settings
        checkpointer = SqliteSaver.from_conn_string(get_settings().checkpoint_db_path).__enter__()
    return graph.compile(checkpointer=checkpointer)
```
Update `_ROUTES`: change `None`/`"start"` to point at `"interviewer_turn"` (not `"interviewer"`).

- [ ] **Step 5: Run the replay test**

Run: `uv run pytest tests/test_interviewer_state.py -v`
Expected: PASS. `p.calls == 5` proves no earlier turn is re-classified on resume.

- [ ] **Step 6: Commit**

```bash
git add sentinel/agents/interviewer.py sentinel/agents/graph.py tests/test_interviewer_state.py
git commit -m "feat(graph): self-looping interviewer_turn node + checkpointed, resumable graph"
```

---

### Task 4: `notify` -> stream-writer everywhere; trainer progress events

**Files:**
- Modify: `sentinel/agents/graph.py` (trainer emits progress)
- Test: `tests/test_interviewer_state.py` or `tests/test_agents.py` (assert custom events)

**Interfaces:**
- Consumes: `get_stream_writer()`; the `_get_writer()` helper from Task 3.
- Produces: trainer emits `{"type":"training","phase":"started"}` before the run and `{"type":"training","phase":"finished"}` after.

- [ ] **Step 1: Write the failing stream test**

```python
def test_interview_defaults_stream_as_custom_events():
    from sentinel.agents.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    p = OneShotProvider(['{"all_defaults": true}'])
    cfg = {"provider_smart": p, "provider_cheap": p, "train_fn": lambda c: None,
           "ticket_dir": "artifacts/tickets"}
    graph = build_graph(checkpointer=MemorySaver())
    thread = {"configurable": {**cfg, "thread_id": "s"}}
    events = []
    for mode, chunk in graph.stream({"event": "start"}, thread, stream_mode=["custom", "updates"]):
        if mode == "custom":
            events.append(chunk)
    # the gate turn interrupts first; resume to accept defaults and flush notices
    from langgraph.types import Command
    for mode, chunk in graph.stream(Command(resume="yes defaults"), thread, stream_mode=["custom", "updates"]):
        if mode == "custom":
            events.append(chunk)
    assert any(e.get("type") == "notify" and "default" in e["text"].lower() for e in events)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_interviewer_state.py::test_interview_defaults_stream_as_custom_events -v`
Expected: FAIL (no custom events, because `_get_writer` returned a no-op under `invoke`; under `stream` it now returns the real writer - if it fails, confirm the writer wiring).

- [ ] **Step 3: Add trainer progress to `trainer_node` in `graph.py`**

```python
def trainer_node(state, config):
    from .interviewer import _get_writer
    writer = _get_writer()
    train_fn = config["configurable"]["train_fn"]
    writer({"type": "training", "phase": "started"})
    try:
        run = train_fn(state["config"])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "event": "run_failed",
                "log": append_log(state, f"trainer: run FAILED ({type(exc).__name__})")}
    writer({"type": "training", "phase": "finished"})
    m = run.result.metrics
    line = f"trainer: run finished, held-out RMSE={m['rmse']:.2f} R2={m['r2']:.3f}"
    return {"train_run": run, "event": "run_finished", "log": append_log(state, line)}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_interviewer_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/graph.py tests/test_interviewer_state.py
git commit -m "feat(graph): stream notify + training progress as custom events"
```

---

### Task 5: CLI driver becomes a stream/resume loop; adapt `test_agents.py`

**Files:**
- Modify: `sentinel/agents/__main__.py`
- Modify: `tests/test_agents.py`

**Interfaces:**
- Consumes: `build_graph`, `graph.stream`, `Command`, `graph.get_state`.
- The scripted-answers demo and `--interactive` both drive the same interrupt path.

- [ ] **Step 1: Rewrite `main()` in `__main__.py`**

Replace the single `graph.invoke` and the `ask` injection with a stream/resume loop. Remove `"ask"` from `configurable`. Keep `SCRIPTED_ANSWERS`.
```python
def main():
    args = _parse_args()
    provider_name = get_settings().sentinel_llm_provider
    print(f"[agent] provider={provider_name}\n")

    configurable = {
        "provider_smart": get_provider("smart"),
        "provider_cheap": get_provider("cheap"),
        "train_fn": run_training,
        "ticket_dir": "artifacts/tickets",
        "thread_id": "cli",
    }
    graph = build_graph()
    thread = {"configurable": configurable}
    answers = None if args.interactive else iter(SCRIPTED_ANSWERS)

    inp = {"event": "start"}
    while True:
        for mode, chunk in graph.stream(inp, thread, stream_mode=["custom", "updates"]):
            if mode == "custom":
                print(f"  {chunk.get('text', chunk)}")
        state = graph.get_state(thread)
        if not state.tasks:                      # no pending interrupt -> graph is done
            break
        prompt = state.tasks[0].interrupts[0].value
        reply = input(prompt + "\n> ") if args.interactive else next(answers, "")
        if not args.interactive:
            print(f"  Q: {prompt}\n  A: {reply}")
        inp = Command(resume=reply)

    final = graph.get_state(thread).values
    # ... existing report/monitor/trace printing unchanged, using `final` ...
```
Keep the existing report/monitor/trace print block, sourcing values from `final`.

- [ ] **Step 2: Adapt `tests/test_agents.py`**

Wherever a test injected `ask`/`notify` and called `graph.invoke`, switch to: build the graph with a `MemorySaver`, add `thread_id`, and drive with the `_drive`-style stream/resume loop (reuse the helper from `tests/test_interviewer_state.py` or inline it). Interview-outcome assertions (collected `config`, defaults, downstream report/monitor) stay identical - only the driver changes. Remove any `notify=` capture in favour of collecting `stream_mode="custom"` events.

- [ ] **Step 3: Run the full agent suite**

Run: `uv run pytest tests/test_agents.py tests/test_interviewer_state.py -v`
Expected: PASS.

- [ ] **Step 4: Smoke-run the CLI offline**

Run: `uv run python -m sentinel.agents 2>&1 | head -40` (uses a real provider only if a key is set; if no key, expect a clean provider error, not a crash in the driver loop).
Expected: the interview streams, scripted answers are consumed, no `ask`/thread errors.

- [ ] **Step 5: Commit**

```bash
git add sentinel/agents/__main__.py tests/test_agents.py
git commit -m "refactor(cli): drive the graph via stream/resume over the interrupt path"
```

---

### Task 6: FastAPI app + SSE endpoints

**Files:**
- Create: `sentinel/api/__init__.py`, `sentinel/api/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `create_app(configurable_factory=None)` returning a FastAPI app. `configurable_factory() -> dict` builds the per-request `configurable` (providers/train_fn/ticket_dir); defaults to the real providers, tests inject fakes.
- Endpoints: `POST /sessions` (-> `{thread_id}` + streams to first interrupt), `POST /sessions/{tid}/resume` (`{"answer": str}`), `GET /sessions/{tid}` (state snapshot).
- SSE event JSON: `{"event": "prompt"|"notify"|"training"|"report"|"done"|"error", "data": {...}}`.

- [ ] **Step 1: Write the failing end-to-end test**

In `tests/test_api.py`:
```python
import json
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver


class OneShotProvider:
    def __init__(self, replies): self._it = iter(replies); self.calls = 0
    def complete(self, messages, **kw): self.calls += 1; return next(self._it)


def _sse_events(resp):
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


def test_session_start_reaches_first_prompt_and_resume_finishes(tmp_path):
    from sentinel.api.app import create_app
    p = OneShotProvider(['{"all_defaults": true}'])
    factory = lambda: {"provider_smart": p, "provider_cheap": p,
                       "train_fn": lambda c: None, "ticket_dir": str(tmp_path)}
    app = create_app(configurable_factory=factory, checkpointer=MemorySaver())
    client = TestClient(app)

    r = client.post("/sessions")
    assert r.status_code == 200
    tid = r.headers["x-thread-id"]
    events = list(_sse_events(r))
    assert events[-1]["event"] == "prompt"          # gate question awaits an answer

    r2 = client.post(f"/sessions/{tid}/resume", json={"answer": "yes defaults"})
    ev2 = list(_sse_events(r2))
    assert any(e["event"] == "notify" for e in ev2)
    assert ev2[-1]["event"] == "done"

    snap = client.get(f"/sessions/{tid}").json()
    assert snap["phase"] == "done"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL (`sentinel.api.app` missing).

- [ ] **Step 3: Implement `sentinel/api/app.py`**

```python
"""FastAPI surface over the resumable agent graph. SSE out, POST in."""
from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from ..agents.graph import build_graph
from ..llm.provider import get_provider
from ..agents.training import run_training


def _default_factory() -> dict:
    return {"provider_smart": get_provider("smart"), "provider_cheap": get_provider("cheap"),
            "train_fn": run_training, "ticket_dir": "artifacts/tickets"}


def _sse(event: str, data) -> str:
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


def create_app(configurable_factory=None, checkpointer=None) -> FastAPI:
    app = FastAPI(title="Sentinel")
    factory = configurable_factory or _default_factory
    graph = build_graph(checkpointer=checkpointer)

    def _thread(tid: str) -> dict:
        return {"configurable": {**factory(), "thread_id": tid}}

    def _run(inp, thread):
        """Stream one graph leg, yielding SSE lines up to the next interrupt/END."""
        for mode, chunk in graph.stream(inp, thread, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield _sse(chunk.get("type", "notify"), chunk)
            elif mode == "updates":
                for node, upd in chunk.items():
                    if isinstance(upd, dict) and upd.get("report"):
                        yield _sse("report", {"text": upd["report"]})
        state = graph.get_state(thread)
        if state.tasks and state.tasks[0].interrupts:
            yield _sse("prompt", {"text": state.tasks[0].interrupts[0].value})
        else:
            yield _sse("done", {"phase": (state.values.get("interview") or {}).get("phase", "done")})

    @app.post("/sessions")
    def start():
        tid = uuid.uuid4().hex
        thread = _thread(tid)
        return StreamingResponse(_run({"event": "start"}, thread),
                                 media_type="text/event-stream", headers={"x-thread-id": tid})

    @app.post("/sessions/{tid}/resume")
    def resume(tid: str, body: dict):
        thread = _thread(tid)
        return StreamingResponse(_run(Command(resume=body.get("answer", "")), thread),
                                 media_type="text/event-stream", headers={"x-thread-id": tid})

    @app.get("/sessions/{tid}")
    def snapshot(tid: str):
        values = graph.get_state(_thread(tid)).values
        prog = values.get("interview") or {}
        return {"phase": prog.get("phase"), "next_prompt": prog.get("next_prompt"),
                "config": getattr(values.get("config"), "__dict__", None),
                "report": values.get("report")}

    return app
```

- [ ] **Step 4: Run the end-to-end test**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke (optional, offline)**

Run: `uv run uvicorn "sentinel.api.app:create_app" --factory --port 8000 &` then `curl -sN -X POST localhost:8000/sessions | head` (needs a provider key for a real run; without one you still get the gate `prompt` since the gate question is emitted before any LLM call). Kill the server after.

- [ ] **Step 6: Commit**

```bash
git add sentinel/api/ tests/test_api.py
git commit -m "feat(api): FastAPI/SSE surface for the resumable agent graph"
```

---

### Task 7: Delete the Streamlit dashboard; update docs; final gate

**Files:**
- Delete: `sentinel/dashboard/`, `tests/test_dashboard_app.py`, `tests/test_dashboard_runner.py`
- Modify: `docs/HANDOFF.md`, `CLAUDE.md` (and `AGENTS.md` if it exists as a real file/symlink), `README.md`

- [ ] **Step 1: Delete the dashboard and its tests**

```bash
git rm -r sentinel/dashboard tests/test_dashboard_app.py tests/test_dashboard_runner.py
```

- [ ] **Step 2: Update the docs**

In `CLAUDE.md`/`AGENTS.md`: replace the interviewer "blocking `ask()` loop / turn-by-turn" convention text with the interrupt-based `interviewer_turn` self-loop + `advance()` state machine, and the run-end-to-end line with the API + CLI stream/resume commands. Note `config["configurable"]` no longer carries `ask`/`notify`; `thread_id` is now required.
In `docs/HANDOFF.md`: move the interrupt/`create_agent`/frontend items from "V1 - next" into "done", and describe the API as the current integration seam (replacing the `GraphRunner` seam sentence). Point the real-frontend note at `POST /sessions` + `resume` + SSE.
In `README.md`: replace dashboard run instructions with `uv run uvicorn "sentinel.api.app:create_app" --factory` and the `python -m sentinel.agents` CLI.

- [ ] **Step 3: Full suite + lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green, no dashboard-import errors, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(v1): retire Streamlit dashboard; docs describe the resumable API"
```

---

## Self-Review

**Spec coverage:**
- interrupt/checkpointer refactor -> Tasks 2, 3. notify->stream -> Task 4. FastAPI/SSE contract (`POST /sessions`, `resume`, `GET /sessions/{id}`) -> Task 6. CLI survives -> Task 5. Delete dashboard + docs -> Task 7. Deps core (not extra) -> Task 1. Replay-safety regression test -> Task 3 (`test_graph_interview_one_llm_call_per_turn`). `create_agent` cut / no auth -> honoured by omission. All spec sections map to a task.

**Placeholder scan:** No TBD/TODO. Task 5 Step 2 and Task 7 Step 2 describe edits to existing files in prose rather than full code, because they are mechanical edits to code the executor has open (adapting an existing driver / doc prose); every net-new unit ships complete code.

**Type consistency:** `InterviewProgress` fields, `start_progress()`, `advance(progress, reply, provider)`, `interviewer_turn(state, config)`, `route_interview`, `build_graph(checkpointer=None)`, `create_app(configurable_factory, checkpointer)`, and the SSE `{"event","data"}` shape are used consistently across tasks. `OneShotProvider` is defined in both new test files (intentional - tasks are independently runnable).

## Open risks to watch during execution

- `graph.get_state(thread).tasks[0].interrupts[0].value` is the documented way to read a pending interrupt's payload in LangGraph 1.2.x; if the accessor differs in 1.2.7, adjust the two call sites (CLI loop, API `_run`) - the shape, not the design, may need a tweak.
- `SqliteSaver.from_conn_string(...)` is a context manager; Task 3 enters it manually for the app-lifetime saver. If that leaks the connection in tests, tests use `MemorySaver` and never hit that path.
