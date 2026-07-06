# Sentinel

A project bridging classical data science and agentic AI: a deterministic ML
core (data prep, feature engineering, AutoML training/eval) wrapped by a
tool-calling data-scientist agent.
The case study is NASA C-MAPSS turbofan **Remaining Useful Life (RUL)**
prediction.

**Milestone 1** is the deterministic DS core - no LLM. It loads the real C-MAPSS
**FD001** subset, engineers rolling-window features, and runs PyCaret AutoML to
train, compare, and evaluate an RUL regression model.

**Milestone 2 (this branch) adds the agent layer**: LangChain's `create_agent`
reasoning hub owns typed data-science tools, a disk-backed model registry, and
confirmation rails with an autonomous override.
See "Agent layer (M2)" below.

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
    provider.py  # config-selected LangChain chat-model factory
  agents/
    agent.py         # create_agent hub + custom autonomy state
    tools.py         # typed DS tools + confirmation rail
    registry.py      # disk-backed, id-addressed model registry
    state.py         # persisted run-configuration shape
    report_writer.py # TrainResult -> plain-language report (grounded, no-fabrication prompt)
    monitor.py       # steps through readings, decides alert/report/mock-action
    domain_context.py # extensible glossary (datasets/metrics) injected into the prompts
    training.py      # full comparison + parameterized retraining adapters
    __main__.py      # guarded/autonomous agent chat loop
  api/
    app.py           # FastAPI/SSE message + confirmation-resume surface
tests/
  test_core_helpers.py   # fast offline unit tests for the pure DS-core helpers
  test_agents.py         # fast offline tests for the agent layer (faked LLM + training)
.env.example                     # template for the .env config (copy to .env, add your key)
docs/learning/01-ds-core.md      # how the DS core fits together (learning note)
docs/learning/04-agentic-ds.md   # reasoning hub, tools, registry, and rails
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

The agent layer replaces the fixed event router with LangChain's
`langchain.agents.create_agent`.
The chat model sees the conversation and typed tool schemas, chooses which tool
to call, receives the result, and continues reasoning until it answers in prose.

The available tools are `save_config`, `train`, `retrain`, `evaluate`,
`compare`, `inspect`, `promote`, `delete`, `write_report`, and `run_monitor`.
Models live in `artifacts/models/` and are addressed by ids such as `et-v1`.
The registry is the single source of truth for metrics, provenance, held-out
readings, and which model is active.

`train`, `retrain`, `promote`, `delete`, and `run_monitor` require confirmation
in guarded mode.
Autonomous mode skips those prompts while still streaming an `auto_approved`
event for every guarded action.
See `docs/learning/04-agentic-ds.md` for the implementation patterns.

### Choosing an LLM provider

The agent receives a LangChain chat model from `sentinel/llm/provider.py`.
Vendor-specific imports remain confined to that file.
Configuration comes from the environment and `.env` through
`sentinel/config.py`, with environment values taking precedence.
Copy the template and fill in your key:

```bash
cp .env.example .env      # then edit .env and set your key
```

`.env` fields (documented in `.env.example`):

| Field                   | Notes                                                                        |
| ----------------------- | ---------------------------------------------------------------------------- |
| `SENTINEL_LLM_PROVIDER` | `groq` (default, free tier - zero API cost) or `anthropic`.                  |
| `GROQ_API_KEY`          | Free Groq key from <https://console.groq.com>. `GROK_API_KEY` also works as an alias. |
| `ANTHROPIC_API_KEY`     | Claude key; only required when the provider is `anthropic`. |
| `SENTINEL_AUTONOMY`     | `guarded` (default) or `autonomous`. |
| `SENTINEL_MODEL_SMART`  | Optional reasoning-model override. |
| `SENTINEL_MODEL_CHEAP`  | Optional report-model override. |

`.env` is gitignored - only `.env.example` is committed, so real keys are never
checked in. Note: **Groq** (groq.com, fast Llama inference - what this app uses)
is a different service from xAI's **Grok**; the key is for Groq, but a habitual
`GROK_API_KEY` spelling is accepted too.

### Chat with the agent

pydantic-settings loads `.env` automatically - no `export` needed:

```bash
uv run python -m sentinel.agents
uv run python -m sentinel.agents --autonomous
```

The default is guarded mode.
The agent converses until it has enough configuration, then uses tools for every
training, comparison, reporting, promotion, deletion, and monitoring action.
Training reuses the M1 pipeline, and monitor alerts write mock maintenance
tickets under `artifacts/tickets/`.

### Run the API

Conversation history and autonomy are checkpointed by LangGraph.
New messages and confirmation replies are separate operations:

```bash
uv run uvicorn "sentinel.api.app:create_app" --factory
```

- `POST /sessions` starts a session and optionally accepts `message` and `autonomy`.
- `POST /sessions/{id}/message` appends a new conversation turn.
- `POST /sessions/{id}/resume` answers pending confirmation ids.
- `GET /sessions/{id}` returns autonomy, the last message, and pending confirmations.

Streams can emit `message`, `tool_call`, `tool_result`, `stage`,
`model_training`, `model_trained`, `confirm`, `auto_approved`, `done`, and
`error`.

**Testing the streaming endpoints: use `curl -N` or a browser `EventSource`, not
Swagger `/docs`.** Swagger buffers the entire `text/event-stream` and only renders it
after the stream closes, so a live run (which holds the connection open for minutes
while PyCaret trains) looks frozen and then dumps everything at once - it is not a
hang. Watch events arrive live with:

```bash
curl -N -X POST localhost:8000/sessions            # note the x-thread-id response header
curl -N -X POST localhost:8000/sessions/<id>/message \
     -H 'content-type: application/json' -d '{"message": "compare et-v1 and et-v2"}'
curl -N -X POST localhost:8000/sessions/<id>/resume \
     -H 'content-type: application/json' -d '{"answer": "yes"}'
```

## Tests, lint, and CI

The unit tests in `tests/` are fast and fully offline.
They use a scripted fake chat model and fake training functions, so there is no
download, PyCaret training run, network access, or live LLM call.

```bash
uv run pytest        # run the tests
uv run ruff check .  # lint (pyflakes + pycodestyle + import order)
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push and pull
request targeting `main`, using `uv sync --locked` against the committed
`uv.lock`.
