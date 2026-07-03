# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Architecture (three layers)

DS core (`sentinel/core/*.py`, M1) -> agent layer (`sentinel/agents/`, `sentinel/llm/`, M2) -> dashboard (later).
The agent layer wraps the DS core and never reaches inside it - it calls the same
`data`/`features`/`automl` functions the M1 `pipeline.py` does.
Design lives in `docs/pdm-agent-design.md`; learning notes in `docs/learning/`.

## Agent layer conventions (M2)

- The graph is a LangGraph `StateGraph` (`sentinel/agents/graph.py`): a hub-and-spoke
  orchestrator that routes purely on `state["event"]` (interview_done, run_finished,
  run_failed, report_ready, monitor_done). Nodes: orchestrator, interviewer, trainer,
  report_writer, monitor.
- **State holds data, dependencies come via `config["configurable"]`** (LangGraph
  `RunnableConfig`), not state. The injected deps are `ask`, `provider_smart`,
  `provider_cheap`, `train_fn`, `ticket_dir`, and `notify` (optional, defaults to
  `print` - the interviewer announces applied defaults through it). This is what lets
  `tests/test_agents.py` run the whole graph offline with fakes - no live LLM, no PyCaret.
- Interviewer is a turn-by-turn chatbot, NOT a batch collector. `run_interview` walks
  `QUESTIONS` and `_resolve_field` resolves ONE field at a time; `classify_turn` makes one
  LLM call per user reply returning `{classification, reply, value, deduced}` where
  classification is CLEAR / UNCLEAR (pushback + offer default same turn) / QUESTION (answer
  from glossary then re-ask, not consumed, not counted) / WANTS_DEFAULT (default this field) /
  ALL_DEFAULTS (short-circuit the rest). Code owns the field order + the `MAX_NONANSWERS`
  bound (fall back to `DEFAULTS`, never loop); LLM owns the language. Each bot message is one
  `ask()` call; a field's ack is prepended to the next question (no end-of-run dump). Don't
  reintroduce batching.
- All-defaults fast path: `run_interview` first offers the `GATE_QUESTION`; `classify_gate`
  (a small yes/no LLM call) short-circuits the whole interview to defaults if the user opts
  in. ALL_DEFAULTS mid-interview does the same for all remaining fields (fixes the bug where
  a global "use defaults for everything" was half-honoured). Both announce each default.
- Deduction (Cognireply-style, confidence-gated at `DEDUCE_CONFIDENCE`=0.6): each turn's
  `deduced` proposals for still-open fields are absorbed (`_absorb_deductions`) only if
  confident + grounded; when the agenda reaches a deduced field it CONFIRMS the value
  (`_CONFIRM`) instead of asking cold. Low confidence -> ask normally. Never invent values.
- LLM access goes through the seam in `sentinel/llm/provider.py` (`Provider` protocol).
  Never import `anthropic`/`groq` outside that file.
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
- Run end to end: `uv run python -m sentinel.agents` (scripted, unattended) or
  `--interactive`. Monitor's mock action writes tickets to `artifacts/tickets/`.
