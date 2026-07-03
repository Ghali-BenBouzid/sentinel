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
docs/learning/01-ds-core.md   # how the pipeline fits together (learning note)
```

## Setup

PyCaret is version-sensitive and needs **Python 3.9-3.11**. The verified set
runs on **Python 3.11**.

```bash
# with uv (recommended - also fetches Python 3.11):
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# or with a system python3.11:
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the direct deps; `requirements.lock.txt` is the full
transitive lock that was actually verified end-to-end.

## Run the M1 pipeline

```bash
.venv/bin/python -m sentinel.pipeline
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
