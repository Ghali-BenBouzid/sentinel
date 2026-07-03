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
  `provider_cheap`, `train_fn`, `ticket_dir`. This is what lets `tests/test_agents.py`
  run the whole graph offline with fakes - no live LLM, no PyCaret.
- LLM access goes through the seam in `sentinel/llm/provider.py` (`Provider` protocol).
  Never import `anthropic`/`groq` outside that file. Provider is env-selected:
  `SENTINEL_LLM_PROVIDER=groq|anthropic` (default groq = free tier); keys read from
  `GROQ_API_KEY` / `ANTHROPIC_API_KEY` env only, never committed.
- Run end to end: `uv run python -m sentinel.agents` (scripted, unattended) or
  `--interactive`. Monitor's mock action writes tickets to `artifacts/tickets/`.
