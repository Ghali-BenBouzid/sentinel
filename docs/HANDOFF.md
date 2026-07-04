# Sentinel - handoff for future sessions

This is an orientation note for a fresh agent (or human) picking up Sentinel.
It says where the project stands, what the MVP delivered, and the proposed V1 / V2 direction.
It references the real artifacts by path rather than repeating them - read those for detail.
Last updated: 2026-07-04.

## Read these first

- `AGENTS.md` (symlinked as `CLAUDE.md`) - build/test commands and the load-bearing conventions. Start here.
- `docs/pdm-agent-design.md` - the agent-layer design and the explicit "out of scope for this milestone" list.
- `docs/learning/01-ds-core.md`, `docs/learning/02-agent-layer.md` - how each layer works and why.
- `docs/superpowers/specs/2026-07-04-dashboard-mvp-design.md` - the dashboard design + a revision note explaining the UI-control-flow rework.
- `docs/superpowers/plans/2026-07-04-dashboard-mvp.md` - the dashboard implementation plan.

## Architecture in one line

DS core (`sentinel/core/`, M1) -> agent layer (`sentinel/agents/`, `sentinel/llm/`, M2) -> dashboard (`sentinel/dashboard/`).
Each layer wraps the one below and never reaches inside it.

## MVP - done and merged

Everything below is on `main` (dashboard work landed via PR #4: https://github.com/Ghali-BenBouzid/sentinel/pull/4).

- **M1 - deterministic DS core.** Loads C-MAPSS FD001, engineers rolling-window features, runs PyCaret AutoML, evaluates the finalized best model on the held-out test set. No LLM. Reference result: Extra Trees, held-out RMSE ~17.1.
- **M2 - agent layer.** A LangGraph hub-and-spoke graph (orchestrator + interviewer / trainer / report_writer / monitor) that drives interview -> train -> report -> monitor. LLM access goes only through the `sentinel/llm/provider.py` seam. Deps are injected via `config["configurable"]`, which is why the whole graph runs offline in tests with fakes.
- **Demo dashboard.** A throwaway Streamlit view (`sentinel/dashboard/`) that runs the real, unchanged M2 graph in the browser. `GraphRunner` (`runner.py`) bridges the graph's blocking `ask`/`notify` seam to the UI via an append-only transcript; `app.py` renders as a pure function of that transcript, held in `st.cache_resource` so it survives reruns and refreshes. Streamlit is an optional extra, not a core dep.

### Lessons already baked in (do not regress these)

- **CV vs held-out test are different measurements.** The leaderboard is cross-validated on the training data (how models are ranked/selected); the headline + report metrics are the winner's held-out test scores (honest generalization, and usually a bit worse). Both are labelled as such in the dashboard and report. Selection stays on cross-validation - do not rank/select by the test set (that is leakage).
- **Keep logic out of the weak model.** The report writer is grounding-constrained (numbers cited verbatim, never derived). The success met/not-met verdict is decided in code (`report_writer._success_verdict`), not by the LLM, because a weak model gets numeric comparisons wrong. Do not move comparisons/derivations back into the prompt.

## V1 - next (agreed direction, not yet built)

Note: there is no formal V1/V2 spec; the split below is synthesized from stated intent and the design doc's post-MVP list. Treat it as a proposal to refine, not a decree. Brainstorm before building.

- **Make the agent layer framework-native with LangGraph `interrupt()` + a checkpointer.** Today the interviewer is a blocking `ask()` loop and the dashboard bridges it with a background thread + queues (`GraphRunner`). The native pattern is: the graph runs until it needs human input, `interrupt()`s (persisting state via a checkpointer), returns control to the UI, and resumes on the next request via `Command(resume=...)`. This removes the thread, the queues, and the injected `ask`/`notify` entirely, and works under any UI. This reworks the M2 interviewer loop, which `AGENTS.md` currently says not to touch - so it is explicitly a V1 change, not a drive-by.
- **Adopt LangChain `create_agent`** to harness the sub-agents more idiomatically instead of hand-wired nodes. Be deliberate: `create_agent` gives a prebuilt ReAct-style loop, while the interviewer is a scripted agenda - decide per sub-agent what becomes a `create_agent` agent vs stays code-driven.
- **Wire the real front end.** A production web front end is being built in a separate, parallel session; the plan is to wire it onto this backend (the `GraphRunner` seam today, the interrupt-based API after the refactor above). Skip Streamlit for production - a real frontend + FastAPI + SSE/WebSocket streaming is the target.

## V2 - vision (later)

- **Genuinely autonomous agents.** Agents that decide for themselves, monitor continuously, and take real actions - beyond the current mock ticket write in `monitor.py`. This milestone needs its own design pass with real safety rails, since an agent taking real maintenance actions is the highest-risk surface.
- **From the design doc's post-MVP list** (`docs/pdm-agent-design.md`): an append-only event stream, real (non-mock) actions, and dataset-agnostic support (today it is FD001-specific).

## Sharp edges / environment gotchas

- **Browser automation does not work in the WSL dev environment** (Claude-in-Chrome extension disconnected; `chrome-devtools-axi` has no Chrome binary). Verify the dashboard end-to-end with Streamlit's `streamlit.testing.v1.AppTest` instead - see `tests/test_dashboard_app.py`.
- **CI runs without the dashboard extra.** `streamlit` is an optional extra (`uv sync --extra dashboard`); the AppTest tests are guarded by `pytest.importorskip("streamlit")` so CI (which runs `uv sync --locked` without the extra) skips them and stays green. Keep it that way.
- **Config is via `.env`** (`SENTINEL_LLM_PROVIDER`, `GROQ_API_KEY`; `GROK_API_KEY` alias accepted). `.env` is gitignored and local-only - it is NOT in any worktree the remote sees, so a fresh checkout has no key. (Never commit the key.)
- **Do not attribute commits or PRs to an AI** (no co-author trailer, no "Generated with..." footer). This is a standing user rule.

## Suggested skills for the next session

- `superpowers:brainstorming` - before building any V1 feature (interrupt refactor, `create_agent` adoption). Design before code.
- `superpowers:writing-plans` then `superpowers:executing-plans` (or `to-issues`) - to break V1 into reviewable steps.
- `superpowers:systematic-debugging` - for any bug or unexpected behavior.
- The Claude/LLM API skill - whenever touching `sentinel/llm/provider.py`, agent prompts, or model choices; do not answer LLM/provider questions from memory.
- Consult `AGENTS.md` conventions on every change to the agent layer.
