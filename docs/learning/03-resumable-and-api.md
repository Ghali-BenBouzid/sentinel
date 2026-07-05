# Learning note 03 - resumable graph, `interrupt()`, and the streaming API

This note explains the V1 refactor: the same M2 agent graph, made resumable across
independent HTTP requests, and exposed as a FastAPI/SSE API.
It assumes you have read note 02 - the sub-agents, the state shape, and the
orchestrator's event-routing table are unchanged.
What changed is *how the interviewer waits for a human* and *how a caller drives the
graph from outside the process*.

If M2 was "one loop of control flow" running inside a single Python process, V1 is
that same loop with every step made independently suspendable, persistable, and
resumable - so the loop can span multiple HTTP requests, multiple processes, even a
server restart, without losing its place.

```
interviewer_turn (self-loop, one interrupt each) -> trainer -> report_writer -> monitor
        ^                                                                        |
        +---- SqliteSaver persists state after every step; Command(resume=...) ----+
                          continues the graph from a POST body
```

---

## 1. The human-in-the-loop problem

A LangGraph node is just a Python function.
The old M2 interviewer (note 02, section 3) got its human input through an injected
`ask(prompt) -> str` callable: call it, and the call blocks the calling thread until
something supplies a reply.
That is fine for a CLI (`input()` blocks the one thread that is the whole program) and
it is fine for a Streamlit demo, where `sentinel/dashboard/runner.py`'s `GraphRunner`
ran the whole graph on a background daemon thread and bridged `ask`/`notify` to the UI
through thread-safe queues (more on why that was retired in section 6).

It is not fine for a web API.
An HTTP request handler has to return a response.
It cannot leave a thread parked inside `ask()` waiting for a human who might reply in
five seconds or five days - that thread (and the memory holding the whole
conversation's state) would have to stay alive for the entire wait, tying up server
capacity per open conversation, unable to survive a redeploy, and unable to be
served by a different process or machine on the next request.

**Resumable** means the opposite of "parked on a thread": the running computation can
be suspended at a well-defined point, its entire state persisted somewhere durable (a
database file, not a stack frame), the process can end completely, and later - on a
completely fresh invocation, possibly in a different process - the same state can be
loaded back and execution continues exactly where it left off.
The conversation stops living in a thread's call stack and starts living in a
checkpoint the graph can rehydrate.
That is the one-line goal of this whole refactor.

---

## 2. `interrupt()` and the replay gotcha

`langgraph.types.interrupt(value)` is LangGraph's suspend point.
Call it inside a node and one of two things happens: if this specific `interrupt()`
call has no saved reply yet, the graph suspends right there - it persists the current
state via the checkpointer and returns control to the caller with `value` recorded as
the pending prompt.
If it *does* have a saved reply (because the caller resumed with
`Command(resume=answer)`), `interrupt()` returns that reply immediately, as if it had
never blocked at all, and the function keeps running.

Here is the gotcha, and it is the crux of the whole refactor: **on resume, the node
function re-executes from the top.**
Not "continues from the interrupt line" - restarts the whole function body.
Every earlier `interrupt()` in that same invocation returns its saved value instantly
(no re-pause), but any other code - an LLM call, a database write, a print statement -
re-runs right along with it.

Concretely: if one node ran a `for field in QUESTIONS: reply = interrupt(...); value =
classify_turn(...)` loop internally (the M2 shape, adapted naively), resuming the
*third* question's interrupt would replay the function from the top: question 1's
`interrupt()` returns its saved reply instantly, but `classify_turn` for question 1
runs again - a second, wasted LLM call, and on the free-tier model a call that is not
even guaranteed to return the same classification twice.
Then question 2's interrupt returns its saved reply, and its `classify_turn` also
replays.
Only question 3's call is genuinely new.
Delivering turn *N* this way replays every one of the *N-1* prior turns' LLM calls, so
the total calls after *N* turns is 1+2+...+*N* - triangular, not linear.

You can watch this happen in five lines with a throwaway repro (no shipped code
touched - just LangGraph's own suspend/resume semantics):

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

calls = 0
def fake_classify(reply):
    global calls
    calls += 1
    return f"value-for-{reply}"

def naive_interview_node(state):
    values = []
    for i in range(3):
        reply = interrupt(f"question {i}")
        values.append(fake_classify(reply))
    return {"values": values}

g = StateGraph(dict)
g.add_node("interview", naive_interview_node)
g.add_edge(START, "interview")
g.add_edge("interview", END)
graph = g.compile(checkpointer=MemorySaver())

thread = {"configurable": {"thread_id": "probe"}}
inp = {}
for ans in ["a", "b", "c"]:
    graph.invoke(inp, thread)
    inp = Command(resume=ans)
result = graph.invoke(inp, thread)
print(calls)  # 6 = 1+2+3, not 3
```

Running this prints `6`, not `3` - confirmed by actually running it while writing this
note.
Three turns delivered, six calls made.
That is the "call-count intuition": if `calls` after *N* turns is *N*, you have one
interrupt per node invocation; if it is *N(N+1)/2*, you have multiple interrupts (or an
LLM call sandwiched between two interrupts) inside one invocation.

**The fix is one-interrupt-per-node.**
`interviewer_turn` in `sentinel/agents/interviewer.py` does exactly one `interrupt()`
and returns immediately after:

```python
def interviewer_turn(state: AgentState, config) -> dict:
    provider = config["configurable"]["provider_smart"]
    progress = state.get("interview") or start_progress()

    reply = interrupt(progress["next_prompt"])   # the ONLY interrupt in this node
    progress = advance(progress, reply, provider)  # the ONLY LLM call, exactly once

    writer = _get_writer()
    for line in progress.get("notices", []):
        writer({"type": "notify", "text": line})

    out = {"interview": progress, "log": append_log(state, "interviewer: turn")}
    if progress["phase"] == "done":
        out["config"] = progress["config"]
        out["event"] = "interview_done"
    return out
```

Each *turn* is now its own graph step, not an iteration inside one function.
`graph.py` wires this with a conditional self-edge:

```python
graph.add_conditional_edges("interviewer_turn", route_interview)
# route_interview: "orchestrator" once interview["phase"] == "done", else "interviewer_turn"
```

So turn 5 is a brand-new node invocation, checkpointed separately from turn 4's
invocation, which has already completed and returned.
Resuming turn 5's interrupt replays *only* turn 5's own function body from the top -
which, before its interrupt, does nothing but read `state.get("interview")` (cheap,
already-checkpointed data).
There is nothing expensive to replay, because nothing expensive ran before the
interrupt.

All the control-flow logic that used to live in the blocking loop (`_resolve_field`,
`run_interview` in the pre-V1 code) now lives in `advance(progress, reply, provider)`
in `interviewer.py` - a pure function, `(InterviewProgress, str, Provider) ->
InterviewProgress`, called exactly once per turn, with no side effects except its
single `provider.complete(...)` call (inside `classify_gate` or `classify_turn`).
It walks the same `phase` state machine note 02 described logically (`gate -> field ->
done`), except now every field of that state machine - `active_index`, `values`,
`deduced`, `resolved`, `history`, `next_prompt`, `nonanswers` - is a plain dict entry in
`InterviewProgress` (`sentinel/agents/state.py`), so it survives being written to a
checkpoint and read back before the next call.

`tests/test_interviewer_state.py::test_graph_interview_one_llm_call_per_turn` is the
regression test that pins this property against the real shipped graph: it drives
`build_graph(checkpointer=MemorySaver())` through a gate decline plus four fields (five
turns total) and asserts `p.calls == 5`.
If someone reintroduces a second interrupt or an LLM call before the interrupt inside
`interviewer_turn`, this test starts failing with a call count above 5 - that is
deliberate; it is the regression test for the exact gotcha this section explains.

---

## 3. Checkpointers: what persists, what `thread_id` keys, how resume continues

A checkpointer is the thing that makes "state survives a suspend" concrete: after
every node returns, LangGraph serializes the *entire* `AgentState` and writes it,
keyed by `thread_id`, to wherever the checkpointer stores it.
When a node is sitting at a pending `interrupt()`, the checkpointer also has the
pending prompt recorded for that thread - that is what `graph.get_state(thread)`
reads back (`state.tasks[0].interrupts[0].value`, used in both the CLI driver and
`api/app.py`'s `_run`).

`thread_id` is the session identifier - one `thread_id` is one independent
conversation/run, with its own checkpoint history.
Tests use `MemorySaver()` (in-process, gone when the test ends).
The real app defaults to `SqliteSaver`, built in `graph.py`'s `_default_checkpointer`:

```python
def _default_checkpointer():
    import sqlite3
    from pathlib import Path
    from langgraph.checkpoint.sqlite import SqliteSaver
    from ..config import get_settings

    path = get_settings().checkpoint_db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # sqlite3 won't create it
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)
```

Two details worth noticing, both there for real reasons:

- **A direct `sqlite3.connect(...)`, not `SqliteSaver.from_conn_string(path)`.**
  `from_conn_string` is a context manager - it is meant to open a connection, hand it
  to you inside a `with` block, and close it when the block exits.
  We want a connection that lives for the whole app process, not one scoped to a
  `with` block, and entering the context manager without a matching, correctly-timed
  exit is fragile: verified against langgraph 1.2.7, a `SqliteSaver.from_conn_string(
  ...).__enter__()` held without keeping the context-manager object itself alive gets
  garbage-collected, and its cleanup closes the connection out from under the app.
  Opening the connection directly (the same thing `from_conn_string` does internally,
  minus the auto-close) sidesteps that entirely.
- **`check_same_thread=False`.**
  FastAPI can serve requests from different threads; a plain sqlite3 connection
  refuses to be touched from any thread but the one that opened it unless you say
  otherwise.

`Command(resume=answer)` is how a caller continues a suspended thread: pass it (instead
of a fresh input dict) as the graph's input, with the same `thread_id` in `config`.
LangGraph looks up that thread's pending interrupt, resumes the suspended node with
`answer` as the return value of its `interrupt()` call, and execution proceeds from
there - onward through `advance()`, back to the orchestrator, or wherever the graph
goes next.

**The serialization constraint.**
Because the *entire* state is serialized after every node - not just at interrupts -
whatever a node puts in `AgentState` has to survive that.
LangGraph 1.2.7's serializer (`JsonPlusSerializer`, msgpack) has no pickle fallback: a
closure, a `pandas.DataFrame`, or a fitted sklearn estimator all raise `Type is not
msgpack serializable` (verified directly - see `tests/test_train_state.py`'s
docstring).
There is one sharp caveat to "native only": a plain `@dataclass` does *not* raise on
the way in.
The serializer quietly encodes it through a deprecated constructor-encoding path (it
warns "Deserializing unregistered type ... will be blocked in a future version. Set
LANGGRAPH_STRICT_MSGPACK=true to block now").
Under `LANGGRAPH_STRICT_MSGPACK=true` (or a future langgraph) that path is blocked and
the value comes back as a bare kwargs dict instead of the dataclass, so any node that
then reads it as an object (`config.failure_threshold`) crashes on a real resume.
That is exactly why the code deliberately stores both `train_state` and the interview
`config` as plain dicts and rehydrates the dataclass locally at each reader, rather
than trusting the dataclass to round-trip.
The pre-V1 code carried a `TrainingRun` - a `predict` closure, a `test_eval`
DataFrame, and a `TrainResult` wrapping a live estimator - straight into
`state["train_run"]`.
None of that is checkpoint-safe.

The fix, in `sentinel/agents/training.py`, is `TrainingRun.to_state()`: it reduces the
run to native-Python data only - a `metrics` dict of floats, the leaderboard as JSON
records (`_records()` round-trips the DataFrame through `json.loads(df.to_json(...))`
specifically to strip numpy scalar types msgpack rejects), the best model's name as a
plain string, `model_path` as a string, and `test_eval` as records.
`trainer_node` (`graph.py`) stores only this dict, in `state["train_state"]`:

```python
cfg = InterviewConfig(**state["config"])  # config crosses as a native dict; rehydrate here
run = train_fn(cfg)
train_state = run.to_state()  # only serializable data crosses the checkpoint boundary
```

The heavy artifacts never round-trip through the checkpoint at all - they are
rehydrated where they are next needed, from disk.
`training.load_predict(model_path)` reloads the persisted PyCaret pipeline and returns
a fresh `predict` function; `report_writer_node` and `monitor_node` both read
`state["train_state"]` and rebuild only what they need (`report_writer_node` builds a
throwaway `TrainResult` with `best_model=None` just to reuse `write_report`'s
signature; `monitor_node` calls `load_predict(ts["model_path"])` and
`pd.DataFrame(ts["test_eval"])`).
`trainer_node` also wraps the whole thing - `train_fn` *and* `to_state()` - in one
`try/except`, because a serialization failure is exactly as fatal to the run as a
training failure, and both need to end up as a clean `run_failed` event instead of a
crash that would truncate an SSE response mid-stream.

---

## 4. Streaming: custom events vs `stream_mode="updates"`

Inside any node, `get_stream_writer()` (from `langgraph.config`) returns a callable;
call it with a dict and that dict is emitted as a **custom** stream event, visible to
anyone consuming `graph.stream(..., stream_mode="custom")`.
This is what replaced the old injected `notify(msg)` callable end to end - applied-
default announcements, per-turn acknowledgements, and the trainer's progress
brackets all go through it now, with no callable threaded through
`config["configurable"]` at all.

`interviewer.py`'s `_get_writer()` wraps the call defensively:

```python
def _get_writer():
    from langgraph.config import get_stream_writer
    try:
        return get_stream_writer()
    except Exception:
        return lambda _e: None
```

`get_stream_writer()` only works inside an active `graph.stream(...)` call; a plain
`graph.invoke(...)` (which most of the offline tests use) has no such context, so the
no-op fallback keeps every node safe to call from either driver.
`trainer_node` uses the same writer to bracket the (potentially long) PyCaret run:
`writer({"type": "training", "phase": "started"})` before, `"finished"` after - so a
caller streaming the run sees it is in progress rather than staring at silence.

The *other* stream mode, `"updates"`, is not custom at all - it surfaces each node's
own return value, keyed by node name, exactly as LangGraph would merge it into state.
`sentinel/api/app.py`'s `_run` generator consumes both modes at once
(`stream_mode=["custom", "updates"]`) and turns them into one SSE vocabulary:

```python
def _run(inp, thread):
    for mode, chunk in graph.stream(inp, thread, stream_mode=["custom", "updates"]):
        if mode == "custom":
            yield _sse(chunk.get("type", "notify"), chunk)
        elif mode == "updates":
            for _node, upd in chunk.items():
                if isinstance(upd, dict) and upd.get("report"):
                    yield _sse("report", {"text": upd["report"]})
    state = graph.get_state(thread)
    if state.tasks and state.tasks[0].interrupts:
        yield _sse("prompt", {"text": state.tasks[0].interrupts[0].value})
    else:
        phase = (state.values.get("interview") or {}).get("phase", "done")
        yield _sse("done", {"phase": phase})
```

Notice the report never needed its own custom event: `report_writer_node`'s ordinary
return value (`{"report": ..., "event": "report_ready", ...}`) already carries the
report text, and `"updates"` mode surfaces it for free.
Only the interviewer's applied-default lines and the trainer's progress brackets
needed the deliberate `writer(...)` calls, because those are things a node wants to
say *without* being a state field a downstream node consumes.

`_sse` itself is the whole SSE encoding:

```python
def _sse(event: str, data) -> str:
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"
```

One JSON object per line, `event`/`data` shape, blank line terminator - that is the
entire Server-Sent-Events wire format this API needs.

### Per-model progress: streaming across the DS-core boundary

The trainer's `training` started/finished bracket has a gap: model comparison is the
slow part (PyCaret cross-validates a whole shelf of regressors), and between the two
brackets the stream goes silent for minutes.
So training also emits **two events per candidate model** - one when it *starts*
(`model_training`) and one when it *finishes* (`model_trained`, with that model's
cross-validated metrics).
Both carry `index` and `total` (1-based, e.g. 3 of 11), so a client can show "training
3 of 11: Extra Trees Regressor..." the instant a model starts, and fill in its score
when it ends - the start event is what keeps the UI from looking frozen during the slow
cross-validation of a single model.

The naming is deliberately PyTorch/Keras-callback style: `on_model_start` and
`on_model_end`, model-level because our loop trains *models* - it has no epochs or
steps (PyCaret does the cross-validation internally), so `on_epoch_*`/`on_step_*` names
would describe machinery that isn't there.
Two purpose-named hooks, not one hook with a `"start"|"end"` phase argument: adding a
lifecycle edge should *add* a hook, not rework the existing one into a phase machine
the caller has to branch on. That is the more additive, more readable design.

The interesting part is *where* the events come from.
The actual training loop lives in the DS core (`sentinel/core/automl.py`), which is
deliberately framework-agnostic - it must not import LangGraph.
So `train_and_evaluate` takes plain `on_model_start(name, index, total)` /
`on_model_end(name, index, total, cv_metrics)` callbacks and knows nothing about
streams; it just trains each model with `create_model` (one at a time, instead of one
opaque `compare_models` call) and calls the hooks around each.
The agent-layer wrapper `run_training` (`sentinel/agents/training.py`) is what bridges
the two worlds: it grabs the active `get_stream_writer()` and hands the DS core a pair
of callbacks that turn each edge into a custom event.

```python
# sentinel/agents/training.py - the bridge
def _model_callbacks():
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:                         # no active stream -> run silently
        return (lambda name, i, n: None), (lambda name, i, n, m: None)

    def on_model_start(name, index, total):
        writer({"type": "model_training", "name": name, "index": index, "total": total})

    def on_model_end(name, index, total, cv_metrics):
        writer({"type": "model_trained", "name": name, "index": index, "total": total, "cv_metrics": cv_metrics})

    return on_model_start, on_model_end
```

That is the general pattern for progress from a layer that should not know about the
graph: the *inner* layer exposes callback seams, the *wrapping node* connects those
seams to the stream.
The DS core stays pure and independently testable (the winner-selection logic is
`_rank_models`, a pure function with its own unit test), and the events still reach the
client with no change to `_run` - because, again, `_run` forwards any `type` field as
its own SSE event.

### Watch out: you cannot see SSE in Swagger `/docs`

A real debugging story worth internalizing.
Swagger UI's "Try it out" **buffers the entire `text/event-stream` and only renders it
after the stream closes.**
Drive a real run through `/docs` and it looks frozen for minutes (while PyCaret trains,
holding the connection open with no bytes flowing), then dumps every event at once when
the graph finishes - which reads exactly like a hang that "finally ended," even though
the server behaved perfectly the whole time.
Test streaming endpoints with something that renders events as they arrive: `curl -N`
(the `-N`/`--no-buffer` flag matters), or a browser `EventSource`.
The `GET /sessions/{id}` snapshot showing `done` while the `POST .../resume` stream
still "hangs" in `/docs` is the same illusion: the graph really did finish and
checkpoint (so the instant JSON snapshot sees `done`), while Swagger is still buffering
the about-to-close stream.

---

## 5. The API contract: three endpoints, SSE out, POST in

`sentinel/api/app.py`'s `create_app` exposes exactly three routes over the same
resumable graph the CLI drives:

- **`POST /sessions`** - starts a new thread (`uuid.uuid4().hex` as `thread_id`),
  streams `{"event": "start"}` through the graph via `_run`, and returns the SSE
  stream with the new `thread_id` in an `x-thread-id` response header.
  The stream runs up to the interviewer's first `interrupt()` (the gate question) and
  stops there with a `prompt` event.
- **`POST /sessions/{tid}/resume`** - takes a JSON body `{"answer": "..."}`, wraps it
  as `Command(resume=body.get("answer", ""))`, and streams the next leg the same way:
  onward to the next `prompt`, or to a `done` event if the graph reached `END`.
- **`GET /sessions/{tid}`** - reads a snapshot straight from the checkpointer
  (`graph.get_state(thread).values`, no execution at all) and returns the interview
  phase, the next prompt, the finished `config` (if any), and the `report` (if any).
  This is the reconnect path: if a client's SSE connection drops mid-stream, it lost
  the *events*, not the *state* - the state was already durably checkpointed - so it
  can just poll this endpoint to see where things stand.

**Why SSE-out/POST-in, not a persistent bidirectional channel (like a WebSocket).**
The conversation is fundamentally turn-based: the client sends exactly one message,
the server streams a bounded burst of reactions to that one message (zero or more
`notify` lines, maybe a `report`), and then either asks exactly one new question or
finishes.
The client never needs to interrupt the server mid-burst, and the server never needs
to push something unprompted between turns - there is nothing happening between
turns at all.
SSE is one-directional server push over a plain HTTP response: perfect for "stream me
this one turn's events, then close the response."
Each new turn is simply a fresh `POST`, so between turns nothing needs to stay open on
either side - all the actual state lives in the checkpointer, not in an open
connection, which is exactly the property this whole refactor was chasing.

---

## 6. What was deleted, and why

The old integration seam, retired in this same effort (`git log --oneline` shows it as
`chore(v1): retire Streamlit dashboard; docs describe the resumable API`), was
`sentinel/dashboard/runner.py`'s `GraphRunner`.
It ran `graph.invoke(...)` on a daemon thread and bridged the graph's *blocking*
`ask()`/`notify()` callables to a Streamlit UI thread through thread-safe queues:
`ask()` blocked until the UI thread pushed a reply into a queue; `notify()` pushed
events into another queue the UI polled.
Streamlit held one `GraphRunner` per session in `st.session_state`/
`st.cache_resource`, alive for as long as that one process ran.

That design's ceiling is exactly the human-in-the-loop problem from section 1: a
thread-per-session model cannot span independent HTTP requests (the thread has to
stay parked in `ask()` across requests, which nothing in a stateless request/response
cycle guarantees), cannot survive a process restart (kill the server, the thread and
everything it was holding vanish with it), and cannot scale horizontally (the thread
and its queues live in one process's memory - a second server instance has no way to
reach them).

| Old (M2 + dashboard) | New (V1) | Why |
|---|---|---|
| injected `ask(q) -> str`, blocks a thread | `interrupt(q)` from `langgraph.types` | suspends and persists via the checkpointer; no thread parked |
| injected `notify(msg)` callback | `get_stream_writer()` custom events | streams out natively, no callable in `configurable` |
| `GraphRunner` (daemon thread + queues) + Streamlit | `SqliteSaver` + `graph.stream(...)` over FastAPI/SSE | resumable statelessly across independent HTTP requests, survives a restart |

Everything else is untouched on purpose: `train_fn`, `provider_smart`,
`provider_cheap`, and `ticket_dir` still come in through
`config["configurable"]`, exactly as note 02 described - that seam was never the
problem, so it was left alone.
Only `ask` and `notify` left `configurable`, because only they were the thread-bound
part.

---

## Try it yourself

Good exercises, in increasing difficulty - each one is runnable against the real
shipped code.

1. **Add a fifth interview field.**
   In `sentinel/agents/interviewer.py`, add a tuple to `QUESTIONS`, e.g.
   `("data_owner", "Who should get the maintenance tickets?")`, plus matching entries
   in `DEFAULTS`, `_DEFAULT_LINE`, and `_CONFIRM` (the same shape as the existing
   `success_metric` entries).
   Run `uv run python -m sentinel.agents --interactive` and watch the new field get
   asked right after `success_metric` - nothing in `advance()`, `interviewer_turn()`,
   or `graph.py` needs to change, because `_ASKED_FIELDS = [f for f, _ in QUESTIONS]`
   derives everything else from the one list.
   You will hit one real gotcha: `InterviewConfig` in `sentinel/agents/state.py` is a
   plain `@dataclass` with a fixed field list, so `InterviewConfig(**values,
   **_EXTRA_DEFAULTS)` raises `TypeError: unexpected keyword argument 'data_owner'`
   until you add `data_owner: str` to the dataclass too - a good reminder that "code
   owns the agenda" means two places, not one.

2. **Break the single-interrupt discipline and watch the replay via call count.**
   This does not touch the shipped interviewer - it is a five-minute standalone
   script that reproduces the exact mechanics section 2 explains, so you feel the
   gotcha instead of taking it on faith.
   Save this as a scratch file and run it with `uv run python`:

   ```python
   from langgraph.checkpoint.memory import MemorySaver
   from langgraph.graph import StateGraph, START, END
   from langgraph.types import interrupt, Command

   calls = 0
   def fake_classify(reply):
       global calls
       calls += 1
       return f"value-for-{reply}"

   def naive_interview_node(state):
       values = []
       for i in range(3):
           reply = interrupt(f"question {i}")
           values.append(fake_classify(reply))
       return {"values": values}

   g = StateGraph(dict)
   g.add_node("interview", naive_interview_node)
   g.add_edge(START, "interview")
   g.add_edge("interview", END)
   graph = g.compile(checkpointer=MemorySaver())

   thread = {"configurable": {"thread_id": "probe"}}
   inp = {}
   for ans in ["a", "b", "c"]:
       graph.invoke(inp, thread)
       inp = Command(resume=ans)
   graph.invoke(inp, thread)
   print(f"calls = {calls}")  # 6, not 3
   ```

   You should see `calls = 6` (1+2+3), not `3` - three turns delivered, six LLM calls
   made, because each resume replays every earlier turn's call inside the same node
   invocation.
   Now compare against
   `tests/test_interviewer_state.py::test_graph_interview_one_llm_call_per_turn`,
   which drives the *real*, self-looping `interviewer_turn` through five turns
   (`uv run pytest tests/test_interviewer_state.py::test_graph_interview_one_llm_call_per_turn -v`)
   and asserts `p.calls == 5` - linear, because each turn is its own node invocation.
   For extra credit, try breaking the real node: put a second `interrupt()` (with an
   LLM call sandwiched between the two) inside `interviewer_turn` itself, then rerun
   that same test and watch the assertion fail with an inflated count - then revert.

3. **Kill and restart a process mid-interview, resume the same `thread_id`.**
   This is the property the whole refactor exists for - and it genuinely works across
   two separate `uv run python` invocations sharing one SQLite file, not just two
   calls in the same process.
   First process (start a thread, then "die" - just let the script exit):

   ```python
   # proc1.py
   import os
   os.environ["CHECKPOINT_DB_PATH"] = "/tmp/restart-demo.sqlite"
   from sentinel.agents.graph import build_graph

   class FakeProvider:
       def complete(self, messages, **kw):
           return '{"all_defaults": false}'

   graph = build_graph()  # real SqliteSaver, not MemorySaver
   thread = {"configurable": {
       "provider_smart": FakeProvider(), "provider_cheap": FakeProvider(),
       "train_fn": lambda c: None, "ticket_dir": "artifacts/tickets",
       "thread_id": "restart-demo",
   }}
   result = graph.invoke({"event": "start"}, thread)
   print(result["__interrupt__"])  # the gate question, pending
   ```

   Second process, run afterwards (a genuinely fresh Python interpreter, fresh
   `SqliteSaver` connection, no shared memory with the first):

   ```python
   # proc2.py
   import os
   os.environ["CHECKPOINT_DB_PATH"] = "/tmp/restart-demo.sqlite"  # same file
   from sentinel.agents.graph import build_graph
   from langgraph.types import Command

   class FakeProvider:
       def complete(self, messages, **kw):
           return '{"classification":"CLEAR","reply":"ok","value":"x","deduced":[]}'

   graph = build_graph()
   thread = {"configurable": {
       "provider_smart": FakeProvider(), "provider_cheap": FakeProvider(),
       "train_fn": lambda c: None, "ticket_dir": "artifacts/tickets",
       "thread_id": "restart-demo",  # SAME thread_id as proc1
   }}
   state = graph.get_state(thread)
   print(state.tasks[0].interrupts[0].value)  # proc1's gate question, read back
   result = graph.invoke(Command(resume="no let's go through it"), thread)
   print(result["__interrupt__"])  # the NEXT question - the interview continued
   ```

   Run `uv run python proc1.py`, then `uv run python proc2.py` (two separate `uv run`
   invocations - nothing kept in memory between them).
   The second process reads back the exact question the first process left pending,
   and resuming moves the interview forward - proof that `SqliteSaver` plus
   `thread_id` is enough, with zero coordination beyond "point at the same file and
   use the same id."

4. **Add a new SSE event type end to end (node writer -> API -> test).**
   Pick a node - `monitor_node` in `sentinel/agents/monitor.py` is a good one, since
   `run_monitor` already loops over readings.
   Add one call to the existing writer plumbing inside `monitor_node` (import
   `_get_writer` from `sentinel.agents.interviewer`, same as `trainer_node` already
   does), e.g. `writer({"type": "monitor_progress", "checked": len(events)})` after
   `run_monitor` returns.
   Now check `sentinel/api/app.py`'s `_run`: `if mode == "custom": yield
   _sse(chunk.get("type", "notify"), chunk)` already forwards *any* `"type"` key as
   its own named SSE event, so a brand-new event type needs **no server-side change
   at all** - that genericness is the point of routing custom events by their own
   `type` field instead of hard-coding each one.
   Prove it end to end the way `tests/test_api.py` does: build a small
   `configurable_factory`, drive `POST /sessions` (and `resume` if needed) with a
   `TestClient`, collect the SSE lines with the test file's `_sse_events(resp)`
   helper, and assert `any(e["event"] == "monitor_progress" for e in events)`.
