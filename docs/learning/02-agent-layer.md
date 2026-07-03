# Learning note 02 - the agent layer

This note explains how the Milestone 2 agent layer fits together and *why* it is
shaped the way it is.
The goal is that you can read `sentinel/agents/*.py` and `sentinel/llm/provider.py`,
understand every decision, and rewrite pieces yourself.
It assumes you have read note 01 - the agent layer wraps the DS core it describes.

If M1 was "one line of data flow", M2 is "one loop of control flow":

```
interview -> train -> report -> monitor
```

A LangGraph graph drives that loop. Everything below is about how.

---

## 1. What a LangGraph `StateGraph` even is

Strip away the vocabulary and a `StateGraph` is three things:

1. **State** - one Python object that flows through the whole run. Here it's
   `AgentState`, a `TypedDict` (`sentinel/agents/state.py`).
2. **Nodes** - plain functions `(state) -> partial_state`. Each returns a *dict of
   the fields it changed*; LangGraph merges that back into the state (last write
   wins). Our nodes are `orchestrator`, `interviewer`, `trainer`, `report_writer`,
   `monitor`.
3. **Edges** - who runs next. A normal edge is "after A, run B". A *conditional*
   edge is "after A, run whatever this function returns" - that's how routing
   works.

You build it, `.compile()` it, then `.invoke(initial_state, config=...)` it. That's
the entire API surface we use. No agent framework, no magic - a graph is just a
state machine whose transitions you wrote.

Run it and watch:

```bash
uv run python -m sentinel.agents      # prints the phases and a TRACE of every node
```

The `TRACE` at the bottom of the output is the `log` field in state - each node
appends one line via `append_log`, so you can literally read the path the run
took through the graph.

---

## 2. The state: what flows between nodes

`AgentState` (in `state.py`) is deliberately small:

```python
class AgentState(TypedDict, total=False):
    event: str            # the wake signal the orchestrator routes on
    config: InterviewConfig
    train_run: Any        # the TrainingRun bundle the trainer produced
    report: str
    alerts: list[dict]
    error: str
    log: list[str]
```

Two design points worth internalizing:

- **State holds *data*, not *dependencies*.** The LLM providers, the training
  function, the "how do I ask a question" callable, and where to write tickets are
  **not** in state. They come in through LangGraph's second argument to a node,
  `config["configurable"]` (a `RunnableConfig`). That keeps the graph a pure state
  machine: the same compiled graph runs against the real Anthropic/Groq providers
  in `__main__.py` and against fakes in the tests, with zero changes to the nodes.
  Look at how `trainer_node(state, config)` pulls `train_fn` out of `config` - that
  indirection is the whole reason the test suite never touches PyCaret.
- **`event` is the important field.** It is the single thing the orchestrator looks
  at to decide what happens next (section 4). Everything else is a by-product a
  sub-agent produced.

---

## 3. Why these four sub-agents (five nodes)

The design calls for an orchestrator plus three sub-agents. Each sub-agent owns
one concern the others don't touch:

- **Interviewer** (`interviewer.py`) - the *only* human-facing node. It turns a
  conversation into an `InterviewConfig`. Nothing else in the graph talks to a
  person.
- **Report writer** (`report_writer.py`) - turns a finished run into prose. Pure
  output; it never trains or monitors.
- **Monitor** (`monitor.py`) - replays held-out readings and decides
  alert/report/act. Pure post-training reaction.

The fifth node, **trainer** (`trainer_node` in `graph.py`), is *not* an LLM
sub-agent - it is the M1 DS core invoked through `training.run_training`. It gets
its own node because "the run finished" and "the run failed" are exactly the
significant events the design says wake the orchestrator. Giving training a node
lets the orchestrator dispatch it and then react to its outcome event, in the same
vocabulary as everything else.

### The interviewer pattern: code owns the agenda, LLM extracts

This is the pattern to copy elsewhere. In `interviewer.py`:

```python
QUESTIONS = [("framing", "..."), ("failure_threshold", "..."), ...]
```

The *code* owns that checklist - the LLM never decides what to ask or in what
order. The human answers in free text; then **one** LLM call (`collect_config`)
turns all the answers into a JSON object we parse into `InterviewConfig`. If the
model returns garbage, `collect_config` falls back to safe defaults rather than
crashing the graph. The LLM does the fuzzy part (free text -> structure) and
nothing else. That's why it's testable with a fake provider that just returns a
fixed JSON string.

---

## 4. Why the orchestrator is "woken by events", not polling

This is the load-bearing design decision, so it's worth being concrete.

The orchestrator is the **hub** of a hub-and-spoke graph. Every sub-agent has an
edge back to it, and it has one conditional edge out, driven by `route`:

```python
_ROUTES = {
    None: "interviewer", "start": "interviewer",
    "interview_done": "trainer",
    "run_finished": "report_writer", "run_failed": "report_writer",
    "report_ready": "monitor",
    "monitor_done": END, "failed_reported": END,
}
```

Each sub-agent, when it finishes, sets `event` to *the significant thing that just
happened* - `interview_done`, `run_finished`, `report_ready`. The orchestrator
routes purely on that field. It never sees, and has nothing to say about,
progress *within* a step (a training epoch, a row being scored). There simply are
no progress ticks in the graph to react to - which is the point. In a real system
the alternative is an orchestrator that polls "are we there yet?"; here the
significant event *is* the wake-up, so routing stays a tiny lookup table.

Trace one full run through `_ROUTES`:

```
START -> orchestrator(event=start)      -> interviewer
      -> orchestrator(interview_done)   -> trainer
      -> orchestrator(run_finished)     -> report_writer
      -> orchestrator(report_ready)     -> monitor
      -> orchestrator(monitor_done)     -> END
```

The failure path forks at the trainer: on an exception, `trainer_node` sets
`event="run_failed"`, the orchestrator still routes to the report writer (so the
failure gets explained), and the report writer sets `failed_reported` -> `END`,
skipping the monitor because there's no model to monitor. One table, both paths.

`route` raises on an unknown event rather than silently stalling - a wrong event is
a bug in a sub-agent, and we want it loud.

---

## 5. The monitor: a simple per-reading threshold (and why nothing fancier)

`monitor.py` replays the FD001 held-out test rows one at a time as if they were
live readings. For each, it predicts RUL and calls `decide`:

```python
def decide(predicted_rul, threshold, warn_factor=2.0):
    if predicted_rul <= threshold:            return "alert"
    if predicted_rul <= threshold * warn_factor: return "report"
    return "ok"
```

`alert` fires the mock action - `_write_ticket` drops a JSON file locally. That's
it. The design explicitly says **do not build a wake-policy system**; a per-reading
threshold is the MVP. `decide` is a pure function precisely so the threshold logic
is unit-testable without a model (see `test_decide_thresholds`), and the actual
prediction is injected (`run.predict`) so tests use a plain lambda instead of
PyCaret.

The threshold itself comes from the interview (`config.failure_threshold`) - the
one place a human number flows all the way through the graph into an action.

---

## 6. How the LLM provider seam plugs in

`sentinel/llm/provider.py` is a seam, not a framework. A `Provider` is anything
with `complete(messages, **kwargs) -> str`. Two implementations - `AnthropicProvider`
(primary) and `GroqProvider` (free tier) - and one selector, `get_provider(tier)`,
that reads `SENTINEL_LLM_PROVIDER` and picks a "smart" or "cheap" model.

Why it matters for the graph:

- **The graph never imports `anthropic` or `groq`.** Nodes receive a `Provider`
  through `config["configurable"]` and call `.complete(...)`. Swap the env var and
  the same graph runs on a different vendor.
- **Cheap vs smart is a real choice, not decoration.** The report writer only has to
  summarize numbers, so it gets the cheap tier; the interviewer has to extract
  structure from messy free text, so it gets the smart tier. `get_provider("cheap")`
  vs `get_provider("smart")` in `__main__.py` is where that's wired.
- **The default is the free tier** (`groq`) so the demo runs at zero API cost - you
  only need a free `GROQ_API_KEY`. Set `SENTINEL_LLM_PROVIDER=anthropic` and
  `ANTHROPIC_API_KEY` to use Claude.

The seam is why the tests can pass a `FakeProvider` that returns a canned string:
the graph asked for "something with `.complete`", and that's all a fake needs to be.

---

## Try it yourself

Good rewrite exercises, in increasing difficulty:

1. **Rebuild the report writer from its docstring.** Delete the *body* of
   `write_report()` in `report_writer.py` and rebuild it using only its docstring
   and the `Provider` protocol - input is a `TrainResult` (`.leaderboard`,
   `.best_model`, `.metrics`), output is text via `provider.complete(...)`. The
   graph wiring around it (`report_writer_node`) does not need to change, and
   `test_write_report_feeds_metrics_to_provider` still has to pass. This is the
   cleanest way to feel how a node and a sub-agent relate.
2. **Add a `best_model_changed` event.** The design lists it as a wake reason.
   Give the trainer a way to detect that this run's best model differs from the last,
   set that event, and add a route for it (e.g. straight to the report writer with a
   "the winning model changed" note). Watch how little of the graph you touch.
3. **Make the monitor smarter without a wake-policy system.** Change `decide` to use
   both predicted *and* actual RUL (the test rows carry the truth), e.g. flag when the
   model is over-optimistic by more than N cycles. Keep it a pure function and add a
   test - that's the bar for "MVP but honest".
