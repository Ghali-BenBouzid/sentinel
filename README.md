# Sentinel

A project bridging classical data science and agentic AI: a deterministic ML
core (data prep, feature engineering, AutoML training/eval) that a later
agentic layer will orchestrate to interview the user, supervise training,
monitor incoming data, and report/act autonomously.
The case study is NASA C-MAPSS turbofan **Remaining Useful Life (RUL)**
prediction.

**Milestone 1** is the deterministic DS core - no LLM. It loads the real C-MAPSS
**FD001** subset, engineers rolling-window features, and runs PyCaret AutoML to
train, compare, and evaluate an RUL regression model.

**Milestone 2 (this branch) adds the agent layer**: a LangGraph graph that wraps
the M1 core - an orchestrator plus interviewer / report-writer / monitor
sub-agents - behind a small LLM provider seam. See "Agent layer (M2)" below.

## Layout

```
sentinel/
  core/
    data.py      # download + load FD001, derive the RUL target
    features.py  # rolling-window (mean/std/slope) feature engineering
    automl.py    # PyCaret train + compare + finalize + evaluate + persist
  pipeline.py    # end-to-end M1 entrypoint
  config.py      # 12-factor settings (pydantic-settings): provider choice + API keys from env/.env
  llm/
    provider.py  # LLM seam: Protocol + AnthropicProvider + GroqProvider (config-selected)
  agents/
    state.py         # graph state + the config the interviewer collects
    graph.py         # the StateGraph: orchestrator routing + node wiring
    interviewer.py   # human-facing sub-agent (code owns agenda, LLM extracts)
    report_writer.py # TrainResult -> plain-language report (grounded, no-fabrication prompt)
    monitor.py       # steps through readings, decides alert/report/mock-action
    domain_context.py # extensible glossary (datasets/metrics) injected into the prompts
    training.py      # thin wrapper that runs the M1 DS core from an InterviewConfig
    __main__.py      # end-to-end runnable: interview -> train -> report -> monitor
tests/
  test_core_helpers.py   # fast offline unit tests for the pure DS-core helpers
  test_agents.py         # fast offline tests for the agent layer (faked LLM + training)
.env.example                     # template for the .env config (copy to .env, add your key)
docs/learning/01-ds-core.md      # how the DS core fits together (learning note)
docs/learning/02-agent-layer.md  # how the agent layer fits together (learning note)
docs/pdm-agent-design.md         # the agent-layer design this milestone implements
.github/workflows/ci.yml         # CI: uv sync + ruff + pytest on pushes/PRs to main
```

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). PyCaret is
version-sensitive and needs **Python 3.9-3.11**; `pyproject.toml` pins the
project to **Python 3.11** (the version verified end-to-end).

```bash
uv sync   # fetches Python 3.11 if needed, creates .venv, installs the locked deps
```

Dependencies are declared in `pyproject.toml` and fully pinned in `uv.lock`, so
`uv sync` reproduces the exact set that was verified.
`uv sync` also installs the dev tooling (`pytest`, `ruff`) declared in the
`dev` dependency group.

## Run the M1 pipeline

```bash
uv run python -m sentinel.pipeline
```

This downloads/caches FD001 under `data/`, builds features, compares ~11 model
families with PyCaret, finalizes the best, evaluates it on the held-out FD001
test set, and writes the model + metrics to `artifacts/`:

- `artifacts/rul_model.pkl` - the finalized RUL model (PyCaret pipeline)
- `artifacts/metrics.json` - best model, held-out RMSE/MAE/R2, full leaderboard
- `artifacts/leaderboard.csv` - the model comparison table

`data/` and `artifacts/` are gitignored (raw data and model binaries are not
committed).

### Reference result

On a clean run (seed 42), the best model is **Extra Trees Regressor** with a
held-out FD001 test score of roughly **RMSE 17.1 / MAE 11.9 / R2 0.82**.

## Agent layer (M2)

The agent layer wraps the M1 core in a LangGraph graph and drives it end to end:
interview the user for config, train, write a plain-language report, then monitor
held-out readings for alerts. See `docs/learning/02-agent-layer.md` for how it
works and `docs/pdm-agent-design.md` for the design.

### Choosing an LLM provider

The graph never imports a vendor SDK - it calls the seam in
`sentinel/llm/provider.py`, which reads its config from `sentinel/config.py`
(pydantic-settings). Config comes from the environment **and** from a `.env`
file (env wins over `.env`). Copy the template and fill in your key:

```bash
cp .env.example .env      # then edit .env and set your key
```

`.env` fields (documented in `.env.example`):

| Field                   | Notes                                                                        |
| ----------------------- | ---------------------------------------------------------------------------- |
| `SENTINEL_LLM_PROVIDER` | `groq` (default, free tier - zero API cost) or `anthropic`.                  |
| `GROQ_API_KEY`          | Free Groq key from <https://console.groq.com>. `GROK_API_KEY` also works as an alias. |
| `ANTHROPIC_API_KEY`     | Claude key (Sonnet for the interviewer, Haiku for the report writer); only if provider is `anthropic`. |

`.env` is gitignored - only `.env.example` is committed, so real keys are never
checked in. Note: **Groq** (groq.com, fast Llama inference - what this app uses)
is a different service from xAI's **Grok**; the key is for Groq, but a habitual
`GROK_API_KEY` spelling is accepted too.

### Run the agent graph end to end

pydantic-settings loads `.env` automatically - no `export` needed:

```bash
uv run python -m sentinel.agents           # scripted interview, runs unattended
uv run python -m sentinel.agents --interactive   # answer the interview yourself
```

This runs the full interview -> train -> report -> monitor flow against FD001,
prints the report and any monitor alerts, and writes mock maintenance tickets to
`artifacts/tickets/`. Training reuses the M1 pipeline, so it downloads/caches
FD001 and runs PyCaret exactly as `python -m sentinel.pipeline` does.

### Run the demo dashboard

A Streamlit dashboard runs the whole agent graph in the browser - chat through the
interview, watch PyCaret train live, read the report, and step through the monitor's
alerts and filed tickets. Streamlit is an optional extra (kept out of the core deps):

```bash
uv sync --extra dashboard
uv run streamlit run sentinel/dashboard/app.py
```

It uses the same provider config as the CLI (`SENTINEL_LLM_PROVIDER` + your key from
`.env`), runs the real graph end to end, and writes the same mock tickets to
`artifacts/tickets/`. Training runs PyCaret live, so the training step takes a few
minutes - the dashboard shows a live status while it compares model families.

## Tests, lint, and CI

The unit tests in `tests/` are fast and fully offline. `test_core_helpers.py`
exercises the pure DS-core helpers (RUL derivation, rolling-window
featurization) on tiny synthetic frames; `test_agents.py` exercises the agent
layer's deterministic parts (provider selection, graph routing, interviewer
extraction, monitor threshold logic, full graph wiring) with the LLM and
training stubbed out - no download, no model training, no live LLM call.

```bash
uv run pytest        # run the tests
uv run ruff check .  # lint (pyflakes + pycodestyle + import order)
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push and pull
request targeting `main`, using `uv sync --locked` against the committed
`uv.lock`.
