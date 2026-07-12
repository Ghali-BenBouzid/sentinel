# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Agent skills

### Issue tracker

Tasks and PRDs use local Markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-role vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Sentinel uses a single root `CONTEXT.md` and system-wide ADRs under `docs/adr/`. See `docs/agents/domain.md`.

## Architecture (three layers)

DS core (`sentinel/core/*.py`, M1) -> agent layer (`sentinel/agents/`, `sentinel/llm/`, M2/V1) -> API (`sentinel/api/`).
The agent layer wraps the DS core and never reaches inside it - it calls the same
`data`/`features`/`automl` functions the M1 `pipeline.py` does.
Design lives in `docs/pdm-agent-design.md`; learning notes in `docs/learning/`.

## Agent layer conventions (M2/V1)

- The graph is a LangGraph `StateGraph` (`sentinel/agents/graph.py`): a hub-and-spoke
  orchestrator that routes purely on `state["event"]` (interview_done, run_finished,
  run_failed, report_ready, monitor_done). Nodes: orchestrator, interviewer_turn, trainer,
  report_writer, monitor.
- **State holds data, dependencies come via `config["configurable"]`** (LangGraph
  `RunnableConfig`), not state. The injected deps are `provider_smart`, `provider_cheap`,
  `train_fn`, `ticket_dir`, and a `thread_id` (now REQUIRED - it addresses the
  checkpoint). There is no `ask`/`notify` in `configurable` any more: the interviewer
  suspends via `interrupt()` and progress/applied-default lines stream out as custom
  events via `get_stream_writer()`. This is what lets `tests/test_agents.py` run the
  whole graph offline with fakes - no live LLM, no PyCaret.
- The interviewer is a self-looping graph node, NOT a blocking `ask()` loop. `interviewer_turn`
  (`sentinel/agents/interviewer.py`) does exactly one `interrupt()` per invocation and loops
  back to itself (`route_interview` in `graph.py`) until the interview is done; the pure
  `advance(progress, reply, provider)` state machine over an `InterviewProgress`
  (`sentinel/agents/state.py`) drives one field at a time and makes one `classify_turn` LLM
  call per resume, returning `{classification, reply, value, deduced}` where classification
  is CLEAR / UNCLEAR (pushback + offer default same turn) / QUESTION (answer from glossary
  then re-ask, not consumed, not counted) / WANTS_DEFAULT (default this field) / ALL_DEFAULTS
  (short-circuit the rest). Code owns the field order + the `MAX_NONANSWERS` bound (fall back
  to `DEFAULTS`, never loop); LLM owns the language. Because each turn is its own node
  invocation, resuming re-executes only that turn - earlier turns are never replayed. Don't
  reintroduce batching or a blocking loop.
- All-defaults fast path: `advance` first offers the `GATE_QUESTION`; `classify_gate`
  (a small yes/no LLM call) short-circuits the whole interview to defaults if the user opts
  in. ALL_DEFAULTS mid-interview does the same for all remaining fields (fixes the bug where
  a global "use defaults for everything" was half-honoured). Both announce each default.
- Deduction (Cognireply-style, confidence-gated at `DEDUCE_CONFIDENCE`=0.6): each turn's
  `deduced` proposals for still-open fields are absorbed (`_absorb_deductions`) only if
  confident + grounded; when the agenda reaches a deduced field it CONFIRMS the value
  (`_CONFIRM`) instead of asking cold. Low confidence -> ask normally. Never invent values.
- LLM access goes through the seam in `sentinel/llm/provider.py` (`Provider` protocol).
  Never import `anthropic`/`groq` outside that file.
- The V2 system prompt separates internal tool execution from user-facing language.
  Suggested next steps must describe achievable outcomes in product language, never tool
  names, argument fields, schemas, invocation instructions, or unsupported capabilities.
  Prompt behavior contracts live in `tests/test_prompts.py`; preserve the grounded examples
  and capability boundary when adding tools.
- Model fallback is credential-aware: `build_agent()` installs `ModelFallbackMiddleware`
  only when the alternate provider has a configured key. Corrective feedback must remain
  deterministic for auth errors and must never raise if its cheap-model fallback is also down.
- **Domain knowledge lives in `sentinel/agents/domain_context.py`** (datasets/metrics
  glossary), not in prompt strings. The report writer and interviewer inject
  `domain_context.glossary()` for grounding. Adding a dataset/metric/model/technique =
  one dict entry, nothing else. The report_writer prompt is deliberately
  grounding-constrained (TIDD-EC system Do/Don't rules: cite only verbatim METRICS
  numbers, never derive/transform - this killed a real "square root of RMSE" fabrication
  on the weak free-tier model). Do not loosen those rules or move numbers out of the
  single METRICS block.
- Config is 12-factor via `sentinel/config.py` (pydantic-settings, `get_settings()` is
  `lru_cache`d): reads env + a `.env` file (env wins). `get_provider` reads it - do not
  read `os.environ` for config elsewhere. Fields: `SENTINEL_LLM_PROVIDER` (groq default),
  `GROQ_API_KEY` (accepts `GROK_API_KEY` alias - captain mistypes "GROK"), `ANTHROPIC_API_KEY`.
  Only `.env.example` is committed; `.env` is gitignored. Tests that change these env vars
  must call `get_settings.cache_clear()` (see the autouse fixtures).
- The graph is resumable: `build_graph()` compiles with a checkpointer (`SqliteSaver` on
  `get_settings().checkpoint_db_path` by default; tests pass a `MemorySaver`). A run
  suspends on `interrupt()` and resumes via `Command(resume=...)` against the same
  `thread_id` - this is what lets the CLI and the API drive the same graph statelessly
  across turns/requests.
- Training results cross state as serializable data, NOT a live object. `trainer_node`
  stores `state["train_state"]` (`TrainingRun.to_state()`, a plain dict), never the
  `TrainingRun` itself - LangGraph 1.2.7's checkpoint serializer has no pickle fallback, so
  a closure/estimator/DataFrame in state would crash on checkpoint. The monitor rehydrates
  a `predict` callable from disk via `training.load_predict(train_state["model_path"])`.
- Run end to end: `uv run python -m sentinel.agents` (scripted, unattended) or
  `--interactive` (both drive the graph via stream/resume over the interrupt path), or the
  API: `uv run uvicorn "sentinel.api.app:create_app" --factory` (`POST /sessions`,
  `POST /sessions/{id}/resume`, `GET /sessions/{id}`, all SSE where relevant - see
  `sentinel/api/app.py`). Monitor's mock action writes tickets to `artifacts/tickets/`.
- Frontend startup hydration must apply only to the thread id found in `localStorage`
  when the page mounts. A thread id returned by `POST /sessions` belongs to the active
  SSE request and must not flow through `selectThread()`, because that function aborts
  in-flight streams before loading a snapshot. Regression coverage is in `web/src/App.test.tsx`.
