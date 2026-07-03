# Sentinel

A project bridging classical data science and agentic AI: a deterministic ML
core (data prep, feature engineering, AutoML training/eval) that a later
agentic layer will orchestrate to interview the user, supervise training,
monitor incoming data, and report/act autonomously.
The case study is NASA C-MAPSS turbofan **Remaining Useful Life (RUL)**
prediction.

**Milestone 1 (this branch) is the deterministic DS core only** - no LLM, no
LangGraph, no dashboard. It loads the real C-MAPSS **FD001** subset, engineers
rolling-window features, and runs PyCaret AutoML to train, compare, and evaluate
an RUL regression model.

## Layout

```
sentinel/
  core/
    data.py      # download + load FD001, derive the RUL target
    features.py  # rolling-window (mean/std/slope) feature engineering
    automl.py    # PyCaret train + compare + finalize + evaluate + persist
  pipeline.py    # end-to-end entrypoint
tests/
  test_core_helpers.py   # fast offline unit tests for the pure DS-core helpers
docs/learning/01-ds-core.md   # how the pipeline fits together (learning note)
.github/workflows/ci.yml      # CI: uv sync + ruff + pytest on pushes/PRs to main
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

## Tests, lint, and CI

The unit tests in `tests/` are fast and fully offline - they exercise the pure
DS-core helpers (RUL derivation, rolling-window featurization) on tiny
synthetic frames, with no download and no model training.

```bash
uv run pytest        # run the tests
uv run ruff check .  # lint (pyflakes + pycodestyle + import order)
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push and pull
request targeting `main`, using `uv sync --locked` against the committed
`uv.lock`.
