# Sentinel - handoff for future sessions

This is an orientation note for a fresh agent (or human) picking up Sentinel.
It says where the project stands, what the MVP delivered, and the proposed V1 / V2 direction.
It references the real artifacts by path rather than repeating them - read those for detail.
Last updated: 2026-07-05.

## Read these first

- `AGENTS.md` (symlinked as `CLAUDE.md`) - build/test commands and the load-bearing conventions. Start here.
- `docs/pdm-agent-design.md` - the agent-layer design and the explicit "out of scope for this milestone" list.
- `docs/learning/01-ds-core.md`, `docs/learning/02-agent-layer.md` - how each layer works and why (note: written for the M2 blocking-`ask()` interviewer, predates the V1 interrupt/`advance()` rework below).
- `docs/superpowers/specs/2026-07-05-v1-resumable-api-design.md` - the V1 design (interrupt/checkpointer, serializable training state, the API).
- `docs/superpowers/plans/2026-07-05-v1-resumable-api.md` - the V1 implementation plan (the dashboard retirement is its Task 7).
- `docs/superpowers/specs/2026-07-04-dashboard-mvp-design.md`, `docs/superpowers/plans/2026-07-04-dashboard-mvp.md` - the retired dashboard's design/plan, kept for history only; the code is deleted.

## Architecture in one line

DS core (`sentinel/core/`, M1) -> agent layer (`sentinel/agents/`, `sentinel/llm/`, M2/V1) -> API (`sentinel/api/`).
Each layer wraps the one below and never reaches inside it.

## MVP - done and merged

Everything below is on `main` (dashboard work landed via PR #4: https://github.com/Ghali-BenBouzid/sentinel/pull/4; the dashboard was later retired, see "V1 - done" below).

- **M1 - deterministic DS core.** Loads C-MAPSS FD001, engineers rolling-window features, runs PyCaret AutoML, evaluates the finalized best model on the held-out test set. No LLM. Reference result: Extra Trees, held-out RMSE ~17.1.
- **M2 - agent layer.** A LangGraph hub-and-spoke graph (orchestrator + interviewer / trainer / report_writer / monitor) that drives interview -> train -> report -> monitor. LLM access goes only through the `sentinel/llm/provider.py` seam. Deps are injected via `config["configurable"]`, which is why the whole graph runs offline in tests with fakes.
- **Demo dashboard (retired in V1).** A throwaway Streamlit view (`sentinel/dashboard/`) that ran the real, unchanged M2 graph in the browser via a `GraphRunner` thread/queue bridge over the blocking `ask`/`notify` seam. It served its purpose (de-risking the graph in a browser) and was deleted once the API below replaced it as the "prove it works" surface.

### Lessons already baked in (do not regress these)

- **CV vs held-out test are different measurements.** The leaderboard is cross-validated on the training data (how models are ranked/selected); the headline + report metrics are the winner's held-out test scores (honest generalization, and usually a bit worse). Both are labelled as such in the report. Selection stays on cross-validation - do not rank/select by the test set (that is leakage).
- **Keep logic out of the weak model.** The report writer is grounding-constrained (numbers cited verbatim, never derived). The success met/not-met verdict is decided in code (`report_writer._success_verdict`), not by the LLM, because a weak model gets numeric comparisons wrong. Do not move comparisons/derivations back into the prompt.

## V1 - done

- **The agent layer is framework-native: LangGraph `interrupt()` + a checkpointer.** The interviewer is no longer a blocking `ask()` loop bridged by a thread + queues. `interviewer_turn` (`sentinel/agents/interviewer.py`) is a self-looping graph node that does one `interrupt()` per invocation over a pure `advance(progress, reply, provider)` state machine (`InterviewProgress` in `state.py`); the graph compiles with a checkpointer (`SqliteSaver` by default, `MemorySaver` in tests) and resumes via `Command(resume=...)` against a `thread_id`. `config["configurable"]` no longer carries `ask`/`notify` - notify/progress are custom stream events via `get_stream_writer()`. Training results cross state as serializable data (`state["train_state"]`, a dict from `TrainingRun.to_state()`), not a live object, since the checkpoint serializer has no pickle fallback; the monitor rehydrates the model from disk via `training.load_predict`.
- **`POST /sessions` / `POST /sessions/{id}/resume` / `GET /sessions/{id}` (SSE) is the current integration seam.** `sentinel/api/app.py` is a FastAPI surface over the same resumable graph the CLI drives - it replaces the `GraphRunner` seam the dashboard used. A client starts a session, streams notify/report/prompt events to the next interrupt, and posts an answer to resume; `GET /sessions/{id}` reads a snapshot straight from the checkpointer so a client can reconnect after losing the SSE stream.
- **The CLI was ported to stream/resume.** `python -m sentinel.agents` (scripted) and `--interactive` drive the same interrupt-based graph via `graph.stream(...)` / `Command(resume=...)`, not a blocking `ask()`.
- **The Streamlit dashboard was retired**, deleted along with its tests, once the API above became the "prove it works" surface.
- **`create_agent` was deliberately deferred to V2** (see below) - not started, not partially done.

## V1 - next / not yet built

- **Wire the real front end.** A production web front end is being built in a separate, parallel session; the plan is to wire it onto the API above (`POST /sessions`, `resume`, SSE) as the integration seam.

## V2 - vision (later)

- **Adopt LangChain `create_agent`** to harness the sub-agents more idiomatically instead of hand-wired nodes. Deliberately deferred out of V1 (not started, not partially done): `create_agent` gives a prebuilt ReAct-style loop, while the interviewer is a scripted agenda, so it needs its own design pass on which sub-agent (if any) becomes a `create_agent` agent vs stays code-driven.
- **Genuinely autonomous agents.** Agents that decide for themselves, monitor continuously, and take real actions - beyond the current mock ticket write in `monitor.py`. This milestone needs its own design pass with real safety rails, since an agent taking real maintenance actions is the highest-risk surface.
- **From the design doc's post-MVP list** (`docs/pdm-agent-design.md`): an append-only event stream, real (non-mock) actions, and dataset-agnostic support (today it is FD001-specific).

## Sharp edges / environment gotchas

- **Browser automation does not work in the WSL dev environment** (Claude-in-Chrome extension disconnected; `chrome-devtools-axi` has no Chrome binary). This mattered for verifying the retired Streamlit dashboard; the API/CLI are verified via their own test suites instead (`tests/test_api.py`, `tests/test_agents.py`) - check actual filenames if picking this up, don't assume.
- **Config is via `.env`** (`SENTINEL_LLM_PROVIDER`, `GROQ_API_KEY`; `GROK_API_KEY` alias accepted). `.env` is gitignored and local-only - it is NOT in any worktree the remote sees, so a fresh checkout has no key. (Never commit the key.)
- **Do not attribute commits or PRs to an AI** (no co-author trailer, no "Generated with..." footer). This is a standing user rule.

## Suggested skills for the next session

- `superpowers:brainstorming` - before building any V2 feature (`create_agent` adoption, autonomous agents). Design before code.
- `superpowers:writing-plans` then `superpowers:executing-plans` (or `to-issues`) - to break V2 into reviewable steps.
- `superpowers:systematic-debugging` - for any bug or unexpected behavior.
- The Claude/LLM API skill - whenever touching `sentinel/llm/provider.py`, agent prompts, or model choices; do not answer LLM/provider questions from memory.
- Consult `AGENTS.md` conventions on every change to the agent layer.
