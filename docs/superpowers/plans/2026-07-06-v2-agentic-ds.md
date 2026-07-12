# V2 Agentic Data Scientist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Sentinel's fixed event-router graph with a LangChain `create_agent` reasoning hub that owns a set of data-science tools (train, retrain-with-params, compare, inspect, report, monitor, promote, delete), backed by a disk model registry and confirmation rails with an autonomous override.

**Architecture:** A `create_agent` (from `langchain.agents`) is the graph hub. Its state is the message history plus an `autonomy` field. All work lives in typed `@tool` functions that wrap the existing DS core / report writer / monitor and read/write a disk-backed model registry. Nothing heavy ever enters graph state - the only cross-checkpoint references are LangChain messages and string model ids. Guarded tools stop for a human `interrupt()` unless the session is autonomous.

**Tech Stack:** Python 3.11, uv, LangChain 1.3.x (`create_agent`, `langchain.agents.middleware.AgentState`), LangGraph 1.2.7 (`InjectedState`, `interrupt`, `Command`, `SqliteSaver`/`MemorySaver`), LangChain chat models (`langchain-groq`, `langchain-anthropic`), PyCaret 3.3.2, FastAPI + SSE, pytest.

## Global Constraints

These bind every task. Copy them into your working memory before each task.

- Python 3.11; managed with `uv`. Run tests with `uv run pytest`, lint with `uv run ruff check .`.
- All unit tests are fully offline: no live LLM, no PyCaret run, no network. Fake the chat model and inject a fake `train_fn`.
- Vendor LLM SDK imports (`anthropic`, `groq`, `langchain_groq`, `langchain_anthropic`) stay confined to `sentinel/llm/provider.py`. No other file imports them.
- Nothing heavy crosses the checkpoint boundary. Graph state holds only LangChain messages, `create_agent`'s bookkeeping keys (`jump_to`, `structured_response`), and the string `autonomy`. No DataFrames, estimators, closures, or dataclasses in state.
- Tools NEVER raise for an expected condition (bad id, malformed args, declined confirmation). They RETURN a descriptive string. The default `ToolNode` re-raises non-validation exceptions and would terminate the graph.
- Guarded tools (stop for confirmation unless autonomous): `train`, `retrain`, `promote`, `delete`, `run_monitor`. Free tools: `save_config`, `evaluate`, `compare`, `inspect`, `write_report`.
- `autonomy` lives in checkpointed graph state and is read in a tool via `Annotated[dict, InjectedState]`. It is set once at session start from the request, defaulting from `get_settings()` (`SENTINEL_AUTONOMY`, default `"guarded"`). It is NEVER read from `config["configurable"]`.
- The agent hub is `create_agent` from `langchain.agents` (LangChain 1.3.x, installed) - the maintained successor to the deprecated `langgraph.prebuilt.create_react_agent`. Do NOT use the deprecated prebuilt.
- The custom state schema subclasses `langchain.agents.middleware.AgentState`, adding only `autonomy: str`. `create_agent` uses `system_prompt=` (str or `SystemMessage`), not `prompt=`; there is no `version` kwarg.
- The registry is the single source of truth for models, addressed by string id. Tools are its only writers.
- Metrics are only comparable within one evaluation config (`rul_cap`, `window`). Comparing across configs must re-evaluate on a common target, never delta raw stored numbers.
- The active model can never be deleted (would dangle `manifest.active`).
- Prose in docs uses plain `-`, never the em dash. In long Markdown, one sentence per line.
- Commit messages: no AI co-author line, no "Generated with" footer.

---

## File Structure

New files:
- `sentinel/agents/registry.py` - disk-backed model registry (Task 1).
- `sentinel/agents/tools.py` - the `confirm()` rail helper + the tool factory `make_tools(...)` (Task 5).
- `sentinel/agents/agent.py` - `DSAgentState`, `SYSTEM_PROMPT`, `build_agent(...)` (Task 6).
- `tests/test_registry.py` (Task 1), `tests/test_train_one.py` (Task 2), `tests/test_provider.py` (Task 4), `tests/test_tools.py` (Task 5), `tests/fakes.py` + `tests/test_agent.py` (Task 6), `tests/test_cli.py` (Task 7), rewritten `tests/test_api.py` (Task 8).

Modified files:
- `sentinel/core/automl.py` - add `train_one` (Task 2).
- `sentinel/llm/provider.py` - rewrite to `get_chat_model` (Task 4).
- `sentinel/config.py` - add model-name override + `sentinel_autonomy` (Task 4).
- `sentinel/agents/report_writer.py` - `write_report` takes a chat model; delete `report_writer_node` (Tasks 3 + 4).
- `sentinel/agents/monitor.py` - keep `decide` / `run_monitor` / `_write_ticket`; delete `monitor_node` + state imports (Task 3).
- `sentinel/agents/state.py` - keep only `InterviewConfig`; delete `AgentState` / `InterviewProgress` / `append_log` (Task 3).
- `sentinel/agents/training.py` - keep `run_training` / `load_predict` / `TrainingRun` / `_stage_event` / `_training_stream`; delete `to_state` (unused after Task 3).
- `sentinel/agents/__main__.py` - rewrite to a chat loop (Task 7).
- `sentinel/api/app.py` - rewrite to message/resume/confirmation surface (Task 8).
- `pyproject.toml` - add chat-model deps (Task 4).

Deleted files:
- `sentinel/agents/interviewer.py` and `sentinel/agents/graph.py` (Task 3).
- `tests/test_interviewer_state.py`, `tests/test_train_state.py` (Task 3); `tests/test_agents.py` is pruned to what survives (Task 3).

---

## Task 1: Model registry

**Files:**
- Create: `sentinel/agents/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `class Registry` constructed as `Registry(models_dir: str | Path)`.
  - `Registry.register(*, family: str, model_path: str | Path, metrics: dict, leaderboard: list[dict], provenance: dict, test_eval: list[dict]) -> str` - copies the saved `.pkl` in, writes `metrics.json` / `provenance.json` / `readings.json`, appends to `manifest.json`, returns the new id (`"<family>-v<N>"`). If no model is active yet, the first registered model becomes active.
  - `Registry.get(model_id: str) -> dict` - returns `{"id", "metrics", "provenance"}`; raises `KeyError` if unknown (callers in Task 5 translate to a string).
  - `Registry.list() -> list[str]`, `Registry.active() -> str | None`, `Registry.set_active(model_id: str) -> None` (raises `KeyError` if unknown).
  - `Registry.remove(model_id: str) -> None` - raises `KeyError` if unknown, `ValueError` if it is the active model.
  - `Registry.readings(model_id: str) -> list[dict]` - the stored `test_eval` records.
  - `Registry.load_predict(model_id: str) -> Callable[[pd.DataFrame], list[float]]` - loads `<id>/model.pkl` via PyCaret (lazy import) and returns a predict fn (same shape as `training.load_predict`).
  - `Registry.provenance(model_id: str) -> dict` - convenience alias for the provenance block (used by `compare` to read `rul_cap`/`window`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
"""The disk-backed model registry: the single source of truth for models."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _fake_saved_model(tmp_path: Path) -> Path:
    """Stand in for a PyCaret .pkl on disk (register only copies bytes)."""
    p = tmp_path / "src_model.pkl"
    p.write_bytes(b"fake-model-bytes")
    return p


def _reg(tmp_path):
    from sentinel.agents.registry import Registry
    return Registry(tmp_path / "models")


def _register(reg, tmp_path, family="et", rul_cap=125, window=5, rmse=17.1):
    return reg.register(
        family=family,
        model_path=_fake_saved_model(tmp_path),
        metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        leaderboard=[{"Model": "Extra Trees", "RMSE": rmse}],
        provenance={"source": "train", "model_id": family, "hyperparameters": {},
                    "config": {"rul_cap": rul_cap, "window": window}, "parent": None},
        test_eval=[{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}],
    )


def test_register_creates_ids_and_layout_and_first_is_active(tmp_path):
    reg = _reg(tmp_path)
    id1 = _register(reg, tmp_path)
    id2 = _register(reg, tmp_path)
    assert id1 == "et-v1" and id2 == "et-v2"        # per-family versioning
    assert reg.active() == "et-v1"                  # first registered auto-activates
    root = tmp_path / "models" / "et-v1"
    assert (root / "model.pkl").read_bytes() == b"fake-model-bytes"
    assert json.loads((root / "metrics.json").read_text())["rmse"] == 17.1
    assert json.loads((root / "provenance.json").read_text())["config"]["rul_cap"] == 125
    assert json.loads((root / "readings.json").read_text())[0]["unit"] == 1
    assert set(reg.list()) == {"et-v1", "et-v2"}


def test_set_active_and_get_and_readings(tmp_path):
    reg = _reg(tmp_path)
    _register(reg, tmp_path)
    id2 = _register(reg, tmp_path)
    reg.set_active(id2)
    assert reg.active() == id2
    assert reg.get(id2)["metrics"]["rmse"] == 17.1
    assert reg.readings(id2)[0]["RUL"] == 40.0
    with pytest.raises(KeyError):
        reg.get("nope")
    with pytest.raises(KeyError):
        reg.set_active("nope")


def test_remove_refuses_active_and_deletes_inactive(tmp_path):
    reg = _reg(tmp_path)
    a = _register(reg, tmp_path)      # active
    b = _register(reg, tmp_path)
    with pytest.raises(ValueError):
        reg.remove(a)                 # active model cannot be deleted
    reg.remove(b)                     # inactive is fine
    assert reg.list() == [a]
    assert not (tmp_path / "models" / b).exists()


def test_load_predict_rehydrates_via_pycaret(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    mid = _register(reg, tmp_path)
    # Fake PyCaret so no real model is loaded.
    import sentinel.agents.registry as R
    monkeypatch.setattr(R, "_pycaret_load_predict",
                        lambda pkl_path: (lambda frame: [50.0] * len(frame)))
    predict = reg.load_predict(mid)
    assert predict(pd.DataFrame([{"s2": 1.0}, {"s2": 2.0}])) == [50.0, 50.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinel.agents.registry'`.

- [ ] **Step 3: Implement `registry.py`**

```python
# sentinel/agents/registry.py
"""Disk-backed model registry: the single source of truth for trained models.

Once retraining exists there are many models (the original winner plus tuned
candidates), so "the current best" / "et-v2" / "the winner" must resolve to
something concrete. This is that something: a small directory per model plus a
manifest naming which one is active.

Layout:
    <models_dir>/
      manifest.json                # {"active": "<id>|null", "models": ["<id>", ...]}
      <id>/
        model.pkl                  # the PyCaret pipeline (copied in on register)
        metrics.json               # {"rmse","mae","r2","leaderboard":[...]}
        provenance.json            # {source, model_id, hyperparameters, config, parent, created_at}
        readings.json              # the held-out monitor readings (test_eval) for THIS config

Only native-JSON data crosses in and out (dicts, lists, strings). The model
itself is rehydrated on demand via `load_predict(id)` (PyCaret, imported lazily),
so nothing heavy is ever handed back to the caller.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


def _pycaret_load_predict(pkl_path: Path) -> Callable[[pd.DataFrame], list[float]]:
    """Load a persisted PyCaret pipeline and return a frame -> predicted-RUL fn.

    Isolated as a module-level function so tests can monkeypatch it without
    touching PyCaret. Mirrors `sentinel.agents.training.load_predict`.
    """
    from pycaret.regression import load_model, predict_model

    model = load_model(str(pkl_path.with_suffix("")))  # load_model wants no .pkl suffix

    def predict(frame: pd.DataFrame) -> list[float]:
        preds = predict_model(model, data=frame)
        col = preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]
        return [float(v) for v in col]

    return predict


class Registry:
    """A directory of trained models plus a manifest of which one is active."""

    def __init__(self, models_dir: str | Path) -> None:
        self.root = Path(models_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        if not self._manifest_path.exists():
            self._write_manifest({"active": None, "models": []})

    # -- manifest helpers -------------------------------------------------
    def _read_manifest(self) -> dict:
        return json.loads(self._manifest_path.read_text())

    def _write_manifest(self, m: dict) -> None:
        self._manifest_path.write_text(json.dumps(m, indent=2))

    def _dir(self, model_id: str) -> Path:
        return self.root / model_id

    def _require(self, model_id: str) -> None:
        if model_id not in self._read_manifest()["models"]:
            raise KeyError(model_id)

    def _next_id(self, family: str) -> str:
        existing = [m for m in self._read_manifest()["models"] if m.rsplit("-v", 1)[0] == family]
        return f"{family}-v{len(existing) + 1}"

    # -- public API -------------------------------------------------------
    def register(
        self,
        *,
        family: str,
        model_path: str | Path,
        metrics: dict,
        leaderboard: list[dict],
        provenance: dict,
        test_eval: list[dict],
    ) -> str:
        """Copy a saved model in, write its json sidecars, return the new id."""
        model_id = self._next_id(family)
        d = self._dir(model_id)
        d.mkdir(parents=True, exist_ok=True)
        src = Path(model_path)
        if src.suffix != ".pkl":
            src = src.with_suffix(".pkl")
        shutil.copyfile(src, d / "model.pkl")
        (d / "metrics.json").write_text(
            json.dumps({**{k: float(v) for k, v in metrics.items()}, "leaderboard": leaderboard}, indent=2)
        )
        prov = {**provenance, "created_at": datetime.now(timezone.utc).isoformat()}
        (d / "provenance.json").write_text(json.dumps(prov, indent=2))
        (d / "readings.json").write_text(json.dumps(test_eval, indent=2))

        m = self._read_manifest()
        m["models"].append(model_id)
        if m["active"] is None:
            m["active"] = model_id  # first model in becomes active
        self._write_manifest(m)
        return model_id

    def get(self, model_id: str) -> dict:
        self._require(model_id)
        metrics = json.loads((self._dir(model_id) / "metrics.json").read_text())
        return {"id": model_id, "metrics": metrics, "provenance": self.provenance(model_id)}

    def provenance(self, model_id: str) -> dict:
        self._require(model_id)
        return json.loads((self._dir(model_id) / "provenance.json").read_text())

    def readings(self, model_id: str) -> list[dict]:
        self._require(model_id)
        return json.loads((self._dir(model_id) / "readings.json").read_text())

    def list(self) -> list[str]:
        return list(self._read_manifest()["models"])

    def active(self) -> str | None:
        return self._read_manifest()["active"]

    def set_active(self, model_id: str) -> None:
        self._require(model_id)
        m = self._read_manifest()
        m["active"] = model_id
        self._write_manifest(m)

    def remove(self, model_id: str) -> None:
        self._require(model_id)
        if self._read_manifest()["active"] == model_id:
            raise ValueError(f"{model_id} is the active model; promote another first")
        shutil.rmtree(self._dir(model_id))
        m = self._read_manifest()
        m["models"].remove(model_id)
        self._write_manifest(m)

    def load_predict(self, model_id: str) -> Callable[[pd.DataFrame], list[float]]:
        self._require(model_id)
        return _pycaret_load_predict(self._dir(model_id) / "model.pkl")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check sentinel/agents/registry.py tests/test_registry.py
git add sentinel/agents/registry.py tests/test_registry.py
git commit -m "feat(registry): disk-backed model registry (single source of truth, id-addressed)"
```

---

## Task 2: Parameterized retraining (`train_one`)

**Files:**
- Modify: `sentinel/core/automl.py`
- Test: `tests/test_train_one.py`

**Interfaces:**
- Consumes: existing `TrainResult`, `_regression_metrics`, `DEFAULT_MODELS` from `sentinel/core/automl.py`.
- Produces: `train_one(model_id: str, hyperparameters: dict, train_df, target: str, test_df, artifacts_dir="artifacts", ignore_features=None, session_id=42, fold=3, on_stage=None) -> TrainResult` - trains ONE named model with the given hyperparameters, finalizes it, evaluates on `test_df`, saves it, and returns a `TrainResult` whose `leaderboard` is a single-row frame for that model. Saves the model to `<artifacts_dir>/retrain_<model_id>.pkl` (a distinct path from `train_and_evaluate`'s `rul_model.pkl`, so the two do not collide) and sets `model_path` to that `.pkl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_one.py
"""train_one: train a single named model with explicit hyperparameters.

PyCaret is faked so the test is offline: we assert train_one wires setup ->
create_model(model_id, **hp) -> finalize -> predict -> save correctly and
returns a one-row leaderboard with held-out metrics.
"""
from __future__ import annotations

import pandas as pd


def test_train_one_trains_named_model_with_hyperparameters(tmp_path, monkeypatch):
    import sentinel.core.automl as A

    calls = {}

    def fake_setup(**kw):
        calls["setup"] = kw
    def fake_create_model(mid, **kw):
        calls["create_model"] = (mid, kw)
        return "FITTED"
    def fake_finalize_model(model):
        calls["finalize"] = model
        return "FINAL"
    def fake_predict_model(model, data):
        # Return the frame with a prediction column PyCaret-style.
        out = data.copy()
        out["prediction_label"] = [40.0] * len(data)
        return out
    def fake_save_model(model, path):
        calls["save"] = path
        # train_one is expected to point model_path at "<path>.pkl"
        (tmp_path / "retrain_et.pkl").write_bytes(b"x")

    monkeypatch.setattr(A, "setup", fake_setup)
    monkeypatch.setattr(A, "create_model", fake_create_model)
    monkeypatch.setattr(A, "finalize_model", fake_finalize_model)
    monkeypatch.setattr(A, "predict_model", fake_predict_model)
    monkeypatch.setattr(A, "save_model", fake_save_model)

    train_df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "s2": [0.1, 0.2], "RUL": [50.0, 49.0]})
    test_df = pd.DataFrame({"unit": [2], "cycle": [10], "s2": [0.3], "RUL": [45.0]})

    stages = []
    result = A.train_one(
        "et", {"n_estimators": 500, "max_depth": 12},
        train_df, target="RUL", test_df=test_df,
        artifacts_dir=str(tmp_path), ignore_features=["unit", "cycle"],
        on_stage=lambda stage, detail="": stages.append(stage),
    )

    assert calls["create_model"] == ("et", {"n_estimators": 500, "max_depth": 12})
    assert calls["setup"]["ignore_features"] == ["unit", "cycle"]
    assert list(result.leaderboard["Model"]) == ["et"]          # single-row leaderboard
    assert set(result.metrics) == {"rmse", "mae", "r2"}         # held-out metrics computed
    assert str(result.model_path).endswith("retrain_et.pkl")
    assert "evaluating" in stages and "saving" in stages        # stage events fired
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_train_one.py -v`
Expected: FAIL with `AttributeError: module 'sentinel.core.automl' has no attribute 'train_one'`.

- [ ] **Step 3: Implement `train_one` in `automl.py`**

Add this function after `train_and_evaluate` (it reuses `_regression_metrics`, `setup`, `create_model`, `finalize_model`, `predict_model`, `save_model`, all already imported at the top of the file):

```python
def train_one(
    model_id: str,
    hyperparameters: dict,
    train_df: pd.DataFrame,
    target: str,
    test_df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    ignore_features: list[str] | None = None,
    session_id: int = 42,
    fold: int = 3,
    on_stage: Callable[..., None] | None = None,
) -> TrainResult:
    """Train ONE named model with explicit hyperparameters, evaluate, and persist.

    The retrain-on-command counterpart to `train_and_evaluate`: instead of
    comparing a shelf and picking a winner, it fits exactly `model_id` with
    `hyperparameters`, finalizes it on the full training data, scores it on the
    held-out `test_df`, and saves it to a distinct path (`retrain_<id>.pkl`) so it
    does not clobber the comparison winner. Returns the same `TrainResult` shape
    with a single-row leaderboard, so downstream (registry, report) is uniform.
    """
    def _stage(name: str, detail: str = "") -> None:
        if on_stage is not None:
            on_stage(name, detail)

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ignore_features = ignore_features or []

    setup(
        data=train_df,
        target=target,
        session_id=session_id,
        ignore_features=ignore_features,
        fold=fold,
        n_jobs=1,
        verbose=False,
    )
    model = create_model(model_id, **hyperparameters)   # trains the one named model
    final_model = finalize_model(model)

    _stage("evaluating")
    preds = predict_model(final_model, data=test_df)
    y_pred = preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]
    metrics = _regression_metrics(test_df[target], y_pred)

    _stage("saving")
    model_path = artifacts_dir / f"retrain_{model_id}"
    save_model(final_model, str(model_path))            # writes retrain_<id>.pkl

    leaderboard = pd.DataFrame([{"Model": model_id, **metrics}])
    return TrainResult(
        leaderboard=leaderboard,
        best_model=final_model,
        metrics=metrics,
        model_path=model_path.with_suffix(".pkl"),
        metrics_path=artifacts_dir / f"retrain_{model_id}_metrics.json",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_train_one.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check sentinel/core/automl.py tests/test_train_one.py
git add sentinel/core/automl.py tests/test_train_one.py
git commit -m "feat(core): train_one - parameterized single-model retraining"
```

---

## Task 3: Tear down the V1 agent graph

This task deletes the fixed-router graph, the interviewer state machine, and the
node wrappers, keeping only the leaf functions the tools will reuse. After it,
the repo has no agent entrypoint yet (rebuilt in Tasks 6-8) but is green: core +
registry + `train_one` + the reused leaf helpers (`write_report`, `decide`,
`run_monitor`, `training.run_training`) all still pass their tests.

**Files:**
- Delete: `sentinel/agents/interviewer.py`, `sentinel/agents/graph.py`, `tests/test_interviewer_state.py`, `tests/test_train_state.py`.
- Modify: `sentinel/agents/state.py` (keep only `InterviewConfig`), `sentinel/agents/report_writer.py` (delete `report_writer_node` + the `AgentState`/`append_log`/`pd`/`Path`/`TrainResult` imports it alone needed), `sentinel/agents/monitor.py` (delete `monitor_node` + `AgentState`/`append_log`/`pd` imports), `sentinel/agents/training.py` (delete unused `to_state`), `sentinel/agents/__main__.py` (temporarily reduce to a stub so the package imports), `sentinel/api/app.py` (temporarily reduce to a stub), `tests/test_agents.py` (prune to surviving tests).

**Interfaces:**
- Consumes: nothing new.
- Produces: `sentinel/agents/state.py` exposing ONLY `InterviewConfig` (the dataclass in Task-6/5 `save_config`). `report_writer.write_report(result, provider, config=None, best_model_name=None)` and `report_writer._success_verdict` unchanged in signature this task (the provider swap is Task 4). `monitor.decide`, `monitor.run_monitor`, `monitor._write_ticket` unchanged. `training.run_training`, `training.load_predict`, `training.TrainingRun` unchanged.

- [ ] **Step 1: Delete the V1 graph, interviewer, and their tests**

```bash
git rm sentinel/agents/interviewer.py sentinel/agents/graph.py \
       tests/test_interviewer_state.py tests/test_train_state.py
```

- [ ] **Step 2: Reduce `state.py` to just `InterviewConfig`**

Replace the whole file body below the module docstring with only the `InterviewConfig` dataclass (delete `InterviewProgress`, `AgentState`, `append_log`). Final content:

```python
"""The structured run config the agent collects before training.

In V1 this file also held the graph's `AgentState`/`InterviewProgress` and a log
helper. Under the V2 agent the graph state is the message history (see
`sentinel/agents/agent.py`), so all that is gone - this file now carries only the
`InterviewConfig` shape that the `save_config` tool persists and the trainer/
report read.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterviewConfig:
    """The structured setup collected before any training runs."""

    framing: str
    failure_threshold: int
    reporting_cadence: str
    success_metric: str
    rul_cap: int = 125
    window: int = 5
```

- [ ] **Step 3: Strip the node wrappers from `report_writer.py` and `monitor.py`**

In `sentinel/agents/report_writer.py`: delete the `report_writer_node` function (lines defining it at the end) and remove the now-unused imports it alone used (`from .state import AgentState, InterviewConfig, append_log` becomes `from .state import InterviewConfig`; keep `pd`, `Path`, `TrainResult` only if `write_report` still uses them - it uses `TrainResult` in its type hint and `result.leaderboard`, keep those; `pd` and `Path` were only used by the node, remove them). Keep `write_report`, `_success_verdict`, `_SYSTEM_PROMPT`, and the word-list constants byte-for-byte.

In `sentinel/agents/monitor.py`: delete `monitor_node`. Remove `from .state import AgentState, append_log` and the `from .training import load_predict` import (the tool will pass a predict fn in), and `import pandas as pd` if no longer used. Keep `decide`, `_write_ticket`, `run_monitor`, `WARN_FACTOR`, `_RUL_UNITS`, and the `domain_context` import.

- [ ] **Step 4: Delete unused `to_state` from `training.py`**

In `sentinel/agents/training.py`, delete the `TrainingRun.to_state` method and the `_records` helper if `to_state` was its only user (check: `_records` is used by `to_state`; grep for other uses - if none, delete both). Keep `run_training`, `load_predict`, `TrainingRun` (as a plain dataclass holding `result`/`test_eval`/`predict`), `_stage_event`, `_STAGE_TEXT`, `_training_stream`.

- [ ] **Step 5: Stub the entrypoints so the package still imports**

Replace `sentinel/agents/__main__.py` with a stub (rebuilt in Task 7):

```python
"""CLI entrypoint - rebuilt as an agent chat loop in Task 7."""
def main() -> None:
    raise SystemExit("The V2 agent CLI is not wired yet (see Task 7).")

if __name__ == "__main__":
    main()
```

Replace `sentinel/api/app.py` with a stub (rebuilt in Task 8):

```python
"""FastAPI surface - rebuilt over the V2 agent in Task 8."""
def create_app(*args, **kwargs):
    raise NotImplementedError("The V2 agent API is not wired yet (see Task 8).")
```

- [ ] **Step 6: Prune `tests/test_agents.py` and `tests/test_api.py`**

`tests/test_agents.py` tests V1 graph routing, the interviewer, provider selection, and `monitor_node` wiring. Delete every test that imports `build_graph`, `interviewer`, `graph`, `monitor_node`, or `get_provider`. KEEP any test that exercises a surviving leaf directly (e.g. a `monitor.decide` threshold test, a `report_writer._success_verdict` test). If nothing survives in a file, `git rm` it. Rewrite `tests/test_api.py` to a single skipped placeholder for now:

```python
# tests/test_api.py
import pytest
pytestmark = pytest.mark.skip(reason="API rebuilt over the V2 agent in Task 8")
```

- [ ] **Step 7: Run the full suite to verify green**

Run: `uv run pytest -v`
Expected: PASS (registry, train_one, core helpers, and any surviving leaf tests). No import errors from the pruned package.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check .
git add -A
git commit -m "refactor(agents): tear down the V1 event-router graph, keep the leaf helpers"
```

---

## Task 4: LLM seam becomes a chat model

**Files:**
- Modify: `pyproject.toml`, `sentinel/llm/provider.py`, `sentinel/config.py`, `sentinel/agents/report_writer.py`
- Test: `tests/test_provider.py`, and a `write_report` test in `tests/test_report_writer.py`

**Interfaces:**
- Consumes: `get_settings()` from `sentinel/config.py`.
- Produces:
  - `get_chat_model(tier: str = "smart", name: str | None = None) -> BaseChatModel` in `sentinel/llm/provider.py`. Returns a `ChatGroq` or `ChatAnthropic` configured from settings; `name` overrides the provider; raises `ValueError` for unknown provider/tier.
  - `write_report(result, chat_model, config=None, best_model_name=None) -> str` now calls `chat_model.invoke([...]).content` instead of `provider.complete([...])`. Message dicts are unchanged (`BaseChatModel.invoke` accepts `[{"role","content"}, ...]`).
  - `Settings` gains `sentinel_autonomy: str = "guarded"` and optional per-tier model overrides `sentinel_model_smart: str | None` / `sentinel_model_cheap: str | None`.

- [ ] **Step 1: Swap the raw SDK deps for the LangChain agent + chat-model deps**

The raw `groq` / `anthropic` SDKs are no longer imported directly (the wrappers pull compatible versions transitively), and V1's `groq>=1.5.0` pin actively conflicts with `langchain-groq` (which needs `groq<1.0.0`). Remove the direct pins, then add the LangChain packages:

```bash
uv remove groq anthropic
uv add langchain langchain-groq langchain-anthropic
```
Expected: `pyproject.toml` drops `groq`/`anthropic` from `dependencies` and gains `langchain` (supplies `create_agent`, Task 6), `langchain-groq`, `langchain-anthropic` (supply the chat models). `uv.lock` resolves to `langchain 1.3.x`, `langchain-groq 1.1.x`, `langchain-anthropic 1.4.x`, with `groq`/`anthropic` now transitive. If any step is already done, `uv` is idempotent.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_provider.py
"""The LLM seam returns a configured LangChain chat model."""
from __future__ import annotations

import pytest


def _clear():
    from sentinel.config import get_settings
    get_settings.cache_clear()


def test_get_chat_model_groq_default(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _clear()
    from langchain_groq import ChatGroq
    from sentinel.llm.provider import get_chat_model
    m = get_chat_model("smart")
    assert isinstance(m, ChatGroq)
    assert m.model_name == "llama-3.3-70b-versatile"


def test_get_chat_model_anthropic(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _clear()
    from langchain_anthropic import ChatAnthropic
    from sentinel.llm.provider import get_chat_model
    assert isinstance(get_chat_model("cheap"), ChatAnthropic)


def test_model_override_from_settings(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SENTINEL_MODEL_SMART", "llama-3.1-8b-instant")
    _clear()
    from sentinel.llm.provider import get_chat_model
    assert get_chat_model("smart").model_name == "llama-3.1-8b-instant"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "bogus")
    _clear()
    from sentinel.llm.provider import get_chat_model
    with pytest.raises(ValueError):
        get_chat_model("smart")
```

```python
# tests/test_report_writer.py
"""write_report drives a chat model (no Provider seam any more)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class FakeChat:
    """Minimal BaseChatModel stand-in: .invoke(messages) -> object with .content."""
    def __init__(self, text): self.text = text
    def invoke(self, messages, **kw):
        class _R: pass
        r = _R(); r.content = self.text
        return r


def test_write_report_uses_chat_model():
    from sentinel.core.automl import TrainResult
    from sentinel.agents.report_writer import write_report
    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}]),
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
        model_path=Path("x"), metrics_path=Path("y"),
    )
    out = write_report(result, FakeChat("Report body."), best_model_name="Extra Trees")
    assert out == "Report body."
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_provider.py tests/test_report_writer.py -v`
Expected: FAIL (`ImportError: cannot import name 'get_chat_model'`; `write_report` still calls `provider.complete`).

- [ ] **Step 4: Rewrite `provider.py`**

```python
"""The LLM provider seam - now a LangChain chat model factory.

Tool-calling needs the richer `bind_tools` interface a plain `complete()->str`
seam cannot express, so the seam returns a LangChain `BaseChatModel`
(`ChatGroq` or `ChatAnthropic`) instead of a custom Provider. Provider choice and
API keys still come from `get_settings()` (env + `.env`), and the vendor SDK
imports stay confined to this file.

Two tiers as before: "smart" (agent reasoning / structure extraction) and "cheap"
(report writing). Either can be overridden by name via settings.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import get_settings

_MODELS = {
    "anthropic": {"smart": "claude-sonnet-5", "cheap": "claude-haiku-4-5"},
    "groq": {"smart": "llama-3.3-70b-versatile", "cheap": "llama-3.1-8b-instant"},
}


def get_chat_model(tier: str = "smart", name: str | None = None) -> BaseChatModel:
    """Build the configured chat model for a `tier` ("smart"/"cheap").

    `name` overrides the configured provider. Per-tier model overrides come from
    settings (`SENTINEL_MODEL_SMART` / `SENTINEL_MODEL_CHEAP`). Raises `ValueError`
    for an unknown provider or tier - fail loud, do not guess.
    """
    settings = get_settings()
    name = (name or settings.sentinel_llm_provider).lower()
    if name not in _MODELS:
        raise ValueError(f"unknown SENTINEL_LLM_PROVIDER {name!r}; expected one of {list(_MODELS)}")
    if tier not in ("smart", "cheap"):
        raise ValueError(f"unknown model tier {tier!r}; expected 'smart' or 'cheap'")
    override = getattr(settings, f"sentinel_model_{tier}", None)
    model = override or _MODELS[name][tier]
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, max_tokens=1024)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=settings.groq_api_key, max_tokens=1024)
```

- [ ] **Step 5: Add config knobs**

In `sentinel/config.py`, add to `Settings` (after `checkpoint_db_path`):

```python
    # V2 agent autonomy: "guarded" (confirm destructive/expensive tools) or
    # "autonomous" (skip confirmations). Session-start default.
    sentinel_autonomy: str = "guarded"

    # Optional per-tier model-name overrides (else the provider default is used).
    sentinel_model_smart: str | None = None
    sentinel_model_cheap: str | None = None
```

- [ ] **Step 6: Point `write_report` at the chat model**

In `sentinel/agents/report_writer.py`, change the final call in `write_report` from:

```python
    return provider.complete(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
```
to:
```python
    return chat_model.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    ).content
```

Rename the parameter `provider` to `chat_model` in the `write_report` signature and docstring. Delete the `from ..llm.provider import Provider` import and the `provider: Provider` type hint (use `chat_model` untyped or `BaseChatModel`). Do NOT touch `_SYSTEM_PROMPT`, `_success_verdict`, or the user-prompt text.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_provider.py tests/test_report_writer.py -v`
Expected: PASS.

- [ ] **Step 8: Full suite, lint, commit**

```bash
uv run pytest
uv run ruff check .
git add -A
git commit -m "feat(llm): swap the Provider seam for a LangChain chat-model factory"
```

---

## Task 5: Tools and the confirmation rail

**Files:**
- Create: `sentinel/agents/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Registry` (Task 1), `train_one` (Task 2), `run_training`/`TrainingRun` (`sentinel/agents/training.py`), `write_report` (Task 4), `monitor.run_monitor`/`monitor.decide`, `InterviewConfig`, `get_stream_writer`, `interrupt`, `InjectedState`.
- Produces:
  - `confirm(action: str, detail: str, autonomy: str) -> str | None` - returns `None` to proceed, or a denial string. Autonomous mode streams `{"type":"auto_approved","tool":action,"detail":detail}` and returns `None`. Guarded mode `interrupt`s and returns a denial string on anything but yes.
  - `make_tools(*, train_fn, chat_model, ticket_dir, registry) -> list` - builds and returns the list of `@tool` callables closed over these deps. `train_fn(config: InterviewConfig) -> TrainingRun` is the injected trainer (the real one is `run_training`; tests inject a fake). `save_config` writes the collected `InterviewConfig` into `registry.root.parent / "config.json"` (a single per-workspace config file) and later tools read it back.

**Design notes for the implementer:**
- Each tool is defined inside `make_tools` as a closure over `registry`, `train_fn`, `chat_model`, `ticket_dir`, so only `autonomy` needs `InjectedState`. Decorate each with `@tool` from `langchain_core.tools`.
- `train` calls `train_fn(config)` -> `TrainingRun`, then `registry.register(family=<winner family>, model_path=run.result.model_path, metrics=run.result.metrics, leaderboard=<records>, provenance={...,"config":{rul_cap,window}}, test_eval=<records>)`. The winner family is derived from the leaderboard's first `Model` via a small name->id map (`_FAMILY = {"Extra Trees Regressor":"et", "Extra Trees":"et", "LightGBM":"lightgbm", ...}`, fallback to a slug of the name).
- `retrain` builds a one-model `TrainingRun` by calling `train_fn` is wrong here - retrain needs `train_one`. So retrain uses a separate injected `retrain_fn(model_id, hyperparameters, rul_cap, window) -> TrainingRun`. To keep injection simple, `make_tools` also takes `retrain_fn`; the real one wraps `run_training`-style plumbing around `train_one`. For this task, `retrain_fn` is injected and faked in tests exactly like `train_fn`.
- `compare` reads both models' `provenance()["config"]`; if equal, delta stored metrics; if different, re-evaluate both via `_evaluate(model_id, rul_cap, window)` on a common config chosen per the Global Constraint, else return a message asking for a config.
- Every tool wraps registry `KeyError` and returns `f"No model '{model_id}' in the registry. Known: {registry.list()}"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
"""The DS tools: registry-backed, rail-guarded, string-returning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class FakeChat:
    def invoke(self, messages, **kw):
        class _R: pass
        r = _R(); r.content = "Report body."
        return r


def _fake_training_run(tmp_path, rmse=17.1, window=5):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult
    pkl = tmp_path / "src.pkl"; pkl.write_bytes(b"m")
    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": rmse}]),
        best_model=object(),
        metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        model_path=pkl, metrics_path=tmp_path / "m.json",
    )
    test_eval = pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}])
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: [1.0])


def _tools(tmp_path):
    from sentinel.agents.registry import Registry
    from sentinel.agents.tools import make_tools
    reg = Registry(tmp_path / "models")
    tools = make_tools(
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda mid, hp, rul_cap, window: _fake_training_run(tmp_path, rmse=16.0),
        chat_model=FakeChat(),
        ticket_dir=str(tmp_path / "tickets"),
        registry=reg,
    )
    return {t.name: t for t in tools}, reg


def _invoke(tool, args, autonomy="autonomous"):
    """Call a structured tool including the InjectedState (autonomy)."""
    return tool.invoke({**args, "state": {"autonomy": autonomy}})


def test_confirm_autonomous_proceeds_and_streams(monkeypatch):
    from sentinel.agents import tools as T
    seen = []
    monkeypatch.setattr(T, "get_stream_writer", lambda: (lambda ev: seen.append(ev)))
    assert T.confirm("promote", "et-v1", "autonomous") is None
    assert seen and seen[0]["type"] == "auto_approved"


def test_confirm_guarded_declined_returns_string(monkeypatch):
    from sentinel.agents import tools as T
    monkeypatch.setattr(T, "interrupt", lambda payload: "no")
    out = T.confirm("promote", "et-v1", "guarded")
    assert isinstance(out, str) and "did not approve" in out


def test_train_registers_winner_and_activates(tmp_path):
    tools, reg = _tools(tmp_path)
    msg = _invoke(tools["train"], {"rul_cap": 125, "window": 5})
    assert reg.active() == "et-v1"
    assert "et-v1" in msg


def test_retrain_registers_candidate(tmp_path):
    tools, reg = _tools(tmp_path)
    _invoke(tools["train"], {})
    _invoke(tools["retrain"], {"model_id": "et", "hyperparameters": {"n_estimators": 500}})
    assert set(reg.list()) == {"et-v1", "et-v2"}


def test_promote_moves_active(tmp_path):
    tools, reg = _tools(tmp_path)
    _invoke(tools["train"], {})
    _invoke(tools["retrain"], {"model_id": "et", "hyperparameters": {}})
    _invoke(tools["promote"], {"model_id": "et-v2"})
    assert reg.active() == "et-v2"


def test_delete_active_refused_with_message(tmp_path):
    tools, reg = _tools(tmp_path)
    _invoke(tools["train"], {})
    out = _invoke(tools["delete"], {"model_id": "et-v1"})
    assert "active" in out.lower()
    assert reg.list() == ["et-v1"]


def test_unknown_model_returns_message_not_raise(tmp_path):
    tools, _ = _tools(tmp_path)
    out = _invoke(tools["evaluate"], {"model_id": "ghost"})
    assert "ghost" in out and "registry" in out.lower()


def test_compare_across_configs_reevaluates(tmp_path, monkeypatch):
    tools, reg = _tools(tmp_path)
    _invoke(tools["train"], {"rul_cap": 125, "window": 5})
    # register a second model with a DIFFERENT config via retrain provenance
    _invoke(tools["retrain"], {"model_id": "et", "hyperparameters": {}, "rul_cap": 100, "window": 5})
    # Fake the re-evaluation path so no PyCaret runs.
    from sentinel.agents import tools as Tmod
    monkeypatch.setattr(Tmod, "_reevaluate", lambda reg, mid, rul_cap, window: {"rmse": 15.0, "mae": 10.0, "r2": 0.9})
    out = _invoke(tools["compare"], {"model_id_a": "et-v1", "model_id_b": "et-v2", "rul_cap": 125, "window": 5})
    assert "et-v1" in out and "et-v2" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL (`No module named 'sentinel.agents.tools'`).

- [ ] **Step 3: Implement `tools.py`**

Implement `confirm`, the `_FAMILY` map, helpers `_records(df)`, `_family_of(name)`, `_reevaluate(registry, model_id, rul_cap, window)` (loads the model via `registry.load_predict`, runs it over the stored readings for that config, returns metrics dict - for the common-config re-eval path; in tests it is monkeypatched), and `make_tools`. Each tool is a `@tool`-decorated closure. Skeleton (fill in every tool; `save_config`, `train`, `retrain`, `evaluate`, `compare`, `inspect`, `promote`, `delete`, `write_report`, `run_monitor`):

```python
"""The data-science tools the agent calls, plus the confirmation rail.

Each tool wraps existing DS-core / report / monitor logic and reads or writes the
model registry. Tools return strings for every outcome, including errors and
declined confirmations - never raise, or the ToolNode terminates the graph. Only
`autonomy` comes from graph state (InjectedState); the registry, trainer, chat
model, and ticket dir are closed over at build time.
"""
from __future__ import annotations

import json
from typing import Annotated, Callable

import pandas as pd
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from ..core import automl
from . import monitor as monitor_mod
from .registry import Registry
from .report_writer import write_report
from .state import InterviewConfig
from .training import TrainingRun

_GUARDED = {"train", "retrain", "promote", "delete", "run_monitor"}
_FAMILY = {"extra trees regressor": "et", "extra trees": "et", "light gradient boosting machine": "lightgbm",
           "lightgbm": "lightgbm", "random forest regressor": "rf", "random forest": "rf"}


def confirm(action: str, detail: str, autonomy: str) -> str | None:
    """Return None to proceed, or a denial string. Never raises."""
    if autonomy == "autonomous":
        get_stream_writer()({"type": "auto_approved", "tool": action, "detail": detail})
        return None
    answer = interrupt({"type": "confirm", "tool": action, "detail": detail})
    if str(answer).strip().lower() in {"y", "yes"}:
        return None
    return f"Declined: the user did not approve {action} ({detail})."


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _family_of(model_name: str) -> str:
    return _FAMILY.get(str(model_name).strip().lower(), str(model_name).strip().lower().replace(" ", "-"))


def _reevaluate(registry: Registry, model_id: str, rul_cap: int, window: int) -> dict:
    """Re-score a model on a common config. Loads the model + its readings and
    predicts. (Monkeypatched in unit tests to stay offline.)"""
    predict = registry.load_predict(model_id)
    readings = pd.DataFrame(registry.readings(model_id))
    preds = predict(readings)
    return automl._regression_metrics(readings["RUL"], list(preds))


def make_tools(*, train_fn, retrain_fn, chat_model, ticket_dir, registry: Registry) -> list:
    config_path = registry.root.parent / "config.json"

    def _load_config() -> InterviewConfig | None:
        if config_path.exists():
            return InterviewConfig(**json.loads(config_path.read_text()))
        return None

    @tool
    def save_config(framing: str, failure_threshold: int, reporting_cadence: str,
                    success_metric: str, rul_cap: int = 125, window: int = 5) -> str:
        """Persist the run configuration gathered from the user."""
        cfg = InterviewConfig(framing, failure_threshold, reporting_cadence, success_metric, rul_cap, window)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg.__dict__, indent=2))
        return f"Saved config: {cfg.__dict__}"

    @tool
    def train(state: Annotated[dict, InjectedState], rul_cap: int = 125, window: int = 5,
              models: list[str] | None = None) -> str:
        """Train and compare the model shelf; register the winner as the active model."""
        denial = confirm("train", f"rul_cap={rul_cap}, window={window}", state["autonomy"])
        if denial:
            return denial
        cfg = _load_config() or InterviewConfig("RUL", 30, "each run", "RMSE under 20", rul_cap, window)
        cfg.rul_cap, cfg.window = rul_cap, window
        run: TrainingRun = train_fn(cfg)
        family = _family_of(run.result.leaderboard.iloc[0]["Model"])
        mid = registry.register(
            family=family, model_path=run.result.model_path, metrics=run.result.metrics,
            leaderboard=_records(run.result.leaderboard),
            provenance={"source": "train", "model_id": family, "hyperparameters": {},
                        "config": {"rul_cap": rul_cap, "window": window}, "parent": None},
            test_eval=_records(run.test_eval),
        )
        m = run.result.metrics
        return f"Trained and registered {mid} (now active): RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} R2={m['r2']:.3f}."

    @tool
    def retrain(model_id: str, hyperparameters: dict, state: Annotated[dict, InjectedState],
                rul_cap: int = 125, window: int = 5) -> str:
        """Retrain one named model with explicit hyperparameters; register it as a candidate."""
        denial = confirm("retrain", f"{model_id} {hyperparameters}", state["autonomy"])
        if denial:
            return denial
        run: TrainingRun = retrain_fn(model_id, hyperparameters, rul_cap, window)
        mid = registry.register(
            family=_family_of(model_id), model_path=run.result.model_path, metrics=run.result.metrics,
            leaderboard=_records(run.result.leaderboard),
            provenance={"source": "retrain", "model_id": model_id, "hyperparameters": hyperparameters,
                        "config": {"rul_cap": rul_cap, "window": window}, "parent": registry.active()},
            test_eval=_records(run.test_eval),
        )
        m = run.result.metrics
        return f"Retrained and registered {mid}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} R2={m['r2']:.3f}."

    @tool
    def evaluate(model_id: str) -> str:
        """Report a registered model's held-out metrics."""
        try:
            info = registry.get(model_id)
        except KeyError:
            return f"No model '{model_id}' in the registry. Known: {registry.list()}"
        m = info["metrics"]
        return f"{model_id}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} R2={m['r2']:.3f}."

    @tool
    def compare(model_id_a: str, model_id_b: str, rul_cap: int | None = None, window: int | None = None) -> str:
        """Compare two models' held-out metrics, re-evaluating on a common config if they differ."""
        try:
            ca = registry.provenance(model_id_a)["config"]
            cb = registry.provenance(model_id_b)["config"]
            ma, mb = registry.get(model_id_a)["metrics"], registry.get(model_id_b)["metrics"]
        except KeyError as e:
            return f"No model {e} in the registry. Known: {registry.list()}"
        if ca != cb:
            if rul_cap is None or window is None:
                active = registry.active()
                if active is None:
                    return ("Cannot compare across different training configs without a common "
                            "rul_cap/window; pass them to compare.")
                common = registry.provenance(active)["config"]
                rul_cap, window = common["rul_cap"], common["window"]
            ma = _reevaluate(registry, model_id_a, rul_cap, window)
            mb = _reevaluate(registry, model_id_b, rul_cap, window)
        d = {k: round(mb[k] - ma[k], 3) for k in ("rmse", "mae", "r2")}
        return (f"{model_id_a}: RMSE={ma['rmse']:.2f} R2={ma['r2']:.3f} | "
                f"{model_id_b}: RMSE={mb['rmse']:.2f} R2={mb['r2']:.3f} | "
                f"delta(b-a): RMSE={d['rmse']} MAE={d['mae']} R2={d['r2']}.")

    @tool
    def inspect(what: str) -> str:
        """Read-only introspection: 'registry' lists models + active; '<id>' shows a model's provenance."""
        if what in ("registry", "models", "list"):
            return f"active={registry.active()} models={registry.list()}"
        try:
            return json.dumps(registry.provenance(what))
        except KeyError:
            return f"No model '{what}' in the registry. Known: {registry.list()}"

    @tool
    def promote(model_id: str, state: Annotated[dict, InjectedState]) -> str:
        """Set a registered model as the active one."""
        denial = confirm("promote", model_id, state["autonomy"])
        if denial:
            return denial
        try:
            registry.set_active(model_id)
        except KeyError:
            return f"No model '{model_id}' in the registry. Known: {registry.list()}"
        return f"Promoted {model_id}; it is now the active model."

    @tool
    def delete(model_id: str, state: Annotated[dict, InjectedState]) -> str:
        """Remove a registered candidate (never the active model)."""
        denial = confirm("delete", model_id, state["autonomy"])
        if denial:
            return denial
        try:
            registry.remove(model_id)
        except KeyError:
            return f"No model '{model_id}' in the registry. Known: {registry.list()}"
        except ValueError as e:
            return str(e)
        return f"Deleted {model_id}."

    @tool
    def write_report_tool() -> str:
        """Write a grounded plain-language report over the active model."""
        active = registry.active()
        if active is None:
            return "No active model to report on; train one first."
        info = registry.get(active)
        result = automl.TrainResult(
            leaderboard=pd.DataFrame(info["metrics"]["leaderboard"]),
            best_model=None, metrics={k: info["metrics"][k] for k in ("rmse", "mae", "r2")},
            model_path=None, metrics_path=None,
        )
        cfg = _load_config()
        return write_report(result, chat_model, cfg, best_model_name=active)

    write_report_tool.name = "write_report"

    @tool
    def run_monitor(state: Annotated[dict, InjectedState]) -> str:
        """Step the active model's stored readings through it and file tickets on alerts."""
        denial = confirm("run_monitor", "file maintenance tickets", state["autonomy"])
        if denial:
            return denial
        active = registry.active()
        if active is None:
            return "No active model to monitor; train one first."
        cfg = _load_config()
        threshold = cfg.failure_threshold if cfg else 30
        predict = registry.load_predict(active)
        readings = pd.DataFrame(registry.readings(active))
        from pathlib import Path
        events = monitor_mod.run_monitor(readings, predict, threshold, Path(ticket_dir))
        alerts = [e for e in events if e["decision"] == "alert"]
        return f"Monitored {len(readings)} readings: {len(events)} flagged, {len(alerts)} alerts filed."

    return [save_config, train, retrain, evaluate, compare, inspect, promote, delete, write_report_tool, run_monitor]
```

Note: `write_report_tool.name = "write_report"` renames the tool so the agent sees `write_report` (the function is named `write_report_tool` only to avoid colliding with the imported `write_report`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: PASS. If the structured-tool `.invoke({..., "state": {...}})` shape rejects the injected `state` key, adjust the test helper to pass state via the documented `InjectedState` test path for the installed LangChain (call the tool's underlying function through `tool.func` with `state=...`), keeping the assertion behavior identical.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check sentinel/agents/tools.py tests/test_tools.py
git add sentinel/agents/tools.py tests/test_tools.py
git commit -m "feat(agents): DS tools + confirmation rail (registry-backed, string-returning)"
```

---

## Task 6: The agent hub

**Files:**
- Create: `sentinel/agents/agent.py`, `tests/fakes.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `make_tools` (Task 5), `Registry` (Task 1), `create_agent` (`from langchain.agents`), `AgentState` (`from langchain.agents.middleware`), `InjectedState`.
- Produces:
  - `class DSAgentState(AgentState): autonomy: str` in `sentinel/agents/agent.py` (base is `langchain.agents.middleware.AgentState`).
  - `SYSTEM_PROMPT: str` - instructs the agent to gather config conversationally then `save_config`, to use tools for all DS work, to confirm nothing itself (the rails do), and injects `domain_context.glossary()`.
  - `build_agent(*, chat_model, train_fn, retrain_fn, tools_chat_model, ticket_dir, models_dir, checkpointer=None) -> CompiledStateGraph` - builds the registry, tools, and returns `create_agent(chat_model, tools, system_prompt=SYSTEM_PROMPT, state_schema=DSAgentState, checkpointer=checkpointer)`. When `checkpointer` is `None`, defaults to a disk `SqliteSaver` (see the CLI note in Task 7).
  - `tests/fakes.py`: `class FakeChatModel` - a scripted `BaseChatModel` emitting a fixed list of `AIMessage`s (with `tool_calls`) in order, with a `bind_tools` passthrough.

- [ ] **Step 1: Write the fake chat model and the failing e2e test**

```python
# tests/fakes.py
"""Offline test doubles for the V2 agent."""
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class FakeChatModel(GenericFakeChatModel):
    """A scripted chat model: yields the given AIMessages in order, ignores tools.

    `bind_tools` is a passthrough so create_agent can bind our tool schemas
    without a real provider. Build with FakeChatModel(messages=iter([...])).
    """
    def bind_tools(self, *args, **kwargs):
        return self
```

```python
# tests/test_agent.py
"""End-to-end agent trajectory, fully offline (scripted model, fake trainer)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from tests.fakes import FakeChatModel


def _fake_training_run(tmp_path, rmse=17.1):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult
    pkl = tmp_path / "src.pkl"; pkl.write_bytes(b"m")
    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": rmse}]),
        best_model=object(), metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        model_path=pkl, metrics_path=tmp_path / "m.json",
    )
    test_eval = pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}])
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: [1.0])


def _tc(name, args, id):
    return {"name": name, "args": args, "id": id}


def _build(tmp_path, scripted, autonomy="autonomous"):
    from sentinel.agents.agent import build_agent
    agent = build_agent(
        chat_model=FakeChatModel(messages=iter(scripted)),
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda mid, hp, rc, w: _fake_training_run(tmp_path, rmse=16.0),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("Report body.")])),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
    )
    return agent


def test_train_retrain_compare_promote_trajectory(tmp_path):
    from sentinel.agents.registry import Registry
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {"rul_cap": 125, "window": 5}, "c1")]),
        AIMessage(content="", tool_calls=[_tc("retrain", {"model_id": "et", "hyperparameters": {"n_estimators": 500}}, "c2")]),
        AIMessage(content="", tool_calls=[_tc("compare", {"model_id_a": "et-v1", "model_id_b": "et-v2"}, "c3")]),
        AIMessage(content="", tool_calls=[_tc("promote", {"model_id": "et-v2"}, "c4")]),
        AIMessage(content="Done: et-v2 is now active."),
    ]
    agent = _build(tmp_path, scripted, autonomy="autonomous")
    th = {"configurable": {"thread_id": "t1"}}
    final = agent.invoke({"messages": [HumanMessage("train, retrain et with 500 trees, compare, promote the winner")],
                          "autonomy": "autonomous"}, th)
    assert "et-v2" in final["messages"][-1].content
    reg = Registry(tmp_path / "models")
    assert reg.active() == "et-v2"
    assert set(reg.list()) == {"et-v1", "et-v2"}


def test_autonomy_persists_across_turns(tmp_path):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
        # second turn: promote WITHOUT resupplying autonomy - must still be autonomous
        AIMessage(content="", tool_calls=[_tc("promote", {"model_id": "et-v1"}, "c2")]),
        AIMessage(content="Promoted."),
    ]
    agent = _build(tmp_path, scripted, autonomy="autonomous")
    th = {"configurable": {"thread_id": "t2"}}
    agent.invoke({"messages": [HumanMessage("train")], "autonomy": "autonomous"}, th)
    final = agent.invoke({"messages": [HumanMessage("promote et-v1")]}, th)  # no autonomy passed
    assert "Promoted" in final["messages"][-1].content   # no interrupt happened -> autonomy persisted


def test_guarded_confirmation_interrupt_and_mapped_resume(tmp_path):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
    ]
    agent = _build(tmp_path, scripted, autonomy="guarded")
    th = {"configurable": {"thread_id": "t3"}}
    agent.invoke({"messages": [HumanMessage("train")], "autonomy": "guarded"}, th)
    st = agent.get_state(th)
    assert st.interrupts and st.interrupts[0].value["tool"] == "train"
    final = agent.invoke(Command(resume={st.interrupts[0].id: "yes"}), th)
    assert "Trained" in final["messages"][-1].content
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL (`No module named 'sentinel.agents.agent'`).

- [ ] **Step 3: Implement `agent.py`**

```python
"""The V2 agent hub: a langchain create_agent reasoning loop over the DS tools.

This replaces V1's fixed event-router graph. The model sees the conversation plus
the tool schemas and decides what to call; the agent's tool node runs the tools;
guarded tools interrupt() for confirmation unless the session is autonomous. State
is the message history plus one `autonomy` field.

`create_agent` (langchain.agents) is the maintained successor to the deprecated
`langgraph.prebuilt.create_react_agent`; we use it directly.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState

from ..config import get_settings
from . import domain_context
from .registry import Registry
from .tools import make_tools


class DSAgentState(AgentState):
    """create_agent's state (messages + its bookkeeping) plus per-session autonomy."""
    autonomy: str


SYSTEM_PROMPT = (
    "You are Sentinel, an autonomous data-scientist agent for predictive maintenance "
    "on NASA C-MAPSS turbofan Remaining-Useful-Life (RUL) prediction.\n\n"
    "You act only through your tools - never claim to have trained, compared, promoted, "
    "or monitored anything except by calling the matching tool and reporting its result.\n"
    "Before the first training run, gather the run configuration conversationally "
    "(what we predict and for what equipment, the RUL failure threshold in cycles, the "
    "reporting cadence, and the success metric), then call save_config. If the user asks "
    "you to just use sensible defaults, call save_config with sensible values and proceed.\n"
    "Do not ask the user to confirm destructive or expensive actions yourself - the system "
    "adds a confirmation step around train, retrain, promote, delete, and run_monitor. Just "
    "call the tool; if it returns a 'Declined' message, respect it and continue.\n"
    "When comparing models, remember metrics are only comparable within one training config "
    "(rul_cap/window); the compare tool handles re-evaluation.\n\n"
    "<glossary>\n" + domain_context.glossary() + "\n</glossary>"
)


def _default_checkpointer():
    """App-lifetime SqliteSaver on the configured path (the V1 pattern).

    `from_conn_string` closes its connection on context exit; open the connection
    directly so it lives as long as the process.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = get_settings().checkpoint_db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_agent(*, chat_model, train_fn, retrain_fn, tools_chat_model, ticket_dir,
                models_dir, checkpointer=None):
    """Assemble the registry, tools, and the create_agent hub."""
    registry = Registry(models_dir)
    tools = make_tools(
        train_fn=train_fn, retrain_fn=retrain_fn,
        chat_model=tools_chat_model, ticket_dir=ticket_dir, registry=registry,
    )
    if checkpointer is None:
        checkpointer = _default_checkpointer()
    return create_agent(
        chat_model, tools, system_prompt=SYSTEM_PROMPT,
        state_schema=DSAgentState, checkpointer=checkpointer,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS (3 tests). The three behaviors are already verified end to end against this exact `create_agent` + `interrupt` + `InjectedState` + mapped-resume stack (see the plan's pre-flight verification). If a version-specific detail around structured-tool `InjectedState` differs, adjust the tool invocation path (not the assertions) until the trajectory, persistence, and interrupt/mapped-resume behaviors hold.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest
uv run ruff check sentinel/agents/agent.py tests/fakes.py tests/test_agent.py
git add sentinel/agents/agent.py tests/fakes.py tests/test_agent.py
git commit -m "feat(agents): create_agent hub with autonomy state + DS tools"
```

---

## Task 7: The CLI chat loop

**Files:**
- Modify: `sentinel/agents/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_agent` (Task 6), `get_chat_model` (Task 4), `run_training` (`training.py`), `get_settings`.
- Produces: `build_real_retrain_fn()` wiring `train_one` into a `TrainingRun`; a `main()` chat loop; a `_run_turn(agent, thread, inp, out=print)` helper that streams events and returns whether a confirmation is pending (kept small and importable so a test can drive it without stdin).

**Design notes:**
- The real `retrain_fn(model_id, hyperparameters, rul_cap, window) -> TrainingRun` mirrors `run_training` but calls `automl.train_one`. Put it in `training.py` as `run_retraining(model_id, hyperparameters, rul_cap, window, data_dir="data", artifacts_dir="artifacts") -> TrainingRun` (loads FD001 at the given rul_cap/window, featurizes, calls `train_one`, builds `test_eval`, returns a `TrainingRun`). This keeps DS plumbing out of the CLI.
- The loop: read a line -> `agent.invoke({"messages":[HumanMessage(line)], "autonomy": mode}, thread)` on the first turn (include autonomy), later turns omit autonomy. After each invoke, check `agent.get_state(thread).interrupts`; if any, prompt `y/n` for each and `agent.invoke(Command(resume={id: answer}), thread)`, repeating until no interrupts; then read the next line.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
"""The CLI turn helper drives the agent and handles pending confirmations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from tests.fakes import FakeChatModel


def _fake_run(tmp_path):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult
    pkl = tmp_path / "s.pkl"; pkl.write_bytes(b"m")
    return TrainingRun(
        result=TrainResult(leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}]),
                           best_model=object(), metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
                           model_path=pkl, metrics_path=tmp_path / "m.json"),
        test_eval=pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}]),
        predict=lambda f: [1.0])


def test_run_turn_completes_without_interrupt_in_autonomous(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.agents.__main__ import run_turn
    agent = build_agent(
        chat_model=FakeChatModel(messages=iter([
            AIMessage(content="", tool_calls=[{"name": "train", "args": {}, "id": "c1"}]),
            AIMessage(content="Trained et-v1."),
        ])),
        train_fn=lambda cfg: _fake_run(tmp_path),
        retrain_fn=lambda *a: _fake_run(tmp_path),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("r")])),
        ticket_dir=str(tmp_path / "t"), models_dir=str(tmp_path / "m"),
        checkpointer=MemorySaver(),
    )
    th = {"configurable": {"thread_id": "cli1"}}
    out = []
    pending = run_turn(agent, th, {"messages": [HumanMessage("train")], "autonomy": "autonomous"}, out.append)
    assert pending is False
    assert any("Trained et-v1" in str(line) for line in out)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (`cannot import name 'run_turn'`).

- [ ] **Step 3: Implement the CLI**

Add `run_retraining` to `training.py` (mirror `run_training`, calling `automl.train_one(model_id, hyperparameters, ...)`). Then rewrite `__main__.py`:

```python
"""Chat with the Sentinel data-scientist agent.

    uv run python -m sentinel.agents                 # guarded (confirm destructive/expensive tools)
    uv run python -m sentinel.agents --autonomous    # delegate: skip confirmations

Provider + key come from SENTINEL_LLM_PROVIDER and your .env, same as before.
"""
from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..config import get_settings
from ..llm.provider import get_chat_model
from .agent import build_agent
from .training import run_retraining, run_training


def run_turn(agent, thread, inp, out=print) -> bool:
    """Run one graph leg, streaming events via `out`. Returns True if a
    confirmation is now pending (caller must resume), else False."""
    for mode, chunk in agent.stream(inp, thread, stream_mode=["custom", "updates"]):
        if mode == "custom":
            out(chunk.get("text", chunk))
        elif mode == "updates":
            for _node, upd in (chunk or {}).items():
                msgs = upd.get("messages") if isinstance(upd, dict) else None
                for m in msgs or []:
                    if getattr(m, "content", ""):
                        out(m.content)
    return bool(agent.get_state(thread).interrupts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autonomous", action="store_true", help="skip confirmations")
    args = parser.parse_args()
    autonomy = "autonomous" if args.autonomous else get_settings().sentinel_autonomy

    agent = build_agent(
        chat_model=get_chat_model("smart"),
        train_fn=run_training,
        retrain_fn=run_retraining,
        tools_chat_model=get_chat_model("cheap"),
        ticket_dir="artifacts/tickets",
        models_dir="artifacts/models",
        checkpointer=None,  # build_agent defaults to a disk SqliteSaver
    )
    thread = {"configurable": {"thread_id": "cli"}}
    print(f"[agent] autonomy={autonomy}. Type your request (Ctrl-D to exit).")
    first = True
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        inp = {"messages": [HumanMessage(line)]}
        if first:
            inp["autonomy"] = autonomy
            first = False
        pending = run_turn(agent, thread, inp)
        while pending:
            st = agent.get_state(thread)
            answers = {}
            for it in st.interrupts:
                v = it.value
                ans = input(f"Confirm {v.get('tool')} ({v.get('detail')})? [y/N] ")
                answers[it.id] = ans
            pending = run_turn(agent, thread, Command(resume=answers))


if __name__ == "__main__":
    main()
```

Note for the implementer: `build_agent(checkpointer=None)` already defaults to a disk `SqliteSaver` (defined in Task 6's `agent.py` as `_default_checkpointer`), so the CLI just passes `checkpointer=None`. Tests keep passing an explicit `MemorySaver()`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest
uv run ruff check sentinel/agents/__main__.py sentinel/agents/training.py tests/test_cli.py
git add -A
git commit -m "feat(cli): agent chat loop with guarded/autonomous modes and confirmation resume"
```

---

## Task 8: The HTTP/SSE surface

**Files:**
- Modify: `sentinel/api/app.py`
- Test: `tests/test_api.py` (replace the Task-3 skip placeholder)

**Interfaces:**
- Consumes: `build_agent` (Task 6), `get_chat_model` (Task 4), `run_training`/`run_retraining` (`training.py`).
- Produces: `create_app(agent_factory=None, checkpointer=None) -> FastAPI` with:
  - `POST /sessions` (body optional `{"autonomy": "...", "message": "..."}`) -> starts a thread, sets autonomy in the first invoke, streams to end/interrupt, returns `x-thread-id`.
  - `POST /sessions/{tid}/message` (`{"message": "..."}`) -> new turn (fresh invoke, no autonomy).
  - `POST /sessions/{tid}/resume` (`{"answers": {id: ans}}` or `{"answer": "y"}` when one pending) -> `Command(resume=...)`.
  - `GET /sessions/{tid}` -> snapshot incl. pending interrupt ids.
  - SSE events: `message`, `tool_call`, `tool_result`, `stage`/`model_training`/`model_trained`, `confirm` (with `interrupt` id), `auto_approved`, `done`, `error`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
"""End-to-end test for the FastAPI/SSE surface over the V2 agent."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from tests.fakes import FakeChatModel


def _fake_run(tmp_path):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult
    pkl = tmp_path / "s.pkl"; pkl.write_bytes(b"m")
    return TrainingRun(
        result=TrainResult(leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}]),
                           best_model=object(), metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
                           model_path=pkl, metrics_path=tmp_path / "m.json"),
        test_eval=pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}]),
        predict=lambda f: [1.0])


def _sse_events(resp):
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


def _app(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=FakeChatModel(messages=iter([
                AIMessage(content="", tool_calls=[{"name": "train", "args": {}, "id": "c1"}]),
                AIMessage(content="Trained et-v1."),
            ])),
            train_fn=lambda cfg: _fake_run(tmp_path),
            retrain_fn=lambda *a: _fake_run(tmp_path),
            tools_chat_model=FakeChatModel(messages=iter([AIMessage("r")])),
            ticket_dir=str(tmp_path / "t"), models_dir=str(tmp_path / "m"),
            checkpointer=checkpointer,
        )
    return create_app(agent_factory=factory, checkpointer=MemorySaver())


def test_autonomous_session_trains_end_to_end(tmp_path):
    client = TestClient(_app(tmp_path))
    r = client.post("/sessions", json={"autonomy": "autonomous", "message": "train"})
    assert r.status_code == 200
    tid = r.headers["x-thread-id"]
    events = list(_sse_events(r))
    assert events[-1]["event"] == "done"
    assert any(e["event"] == "message" and "Trained" in e["data"].get("text", "") for e in events)


def test_guarded_session_emits_confirm_with_interrupt_id(tmp_path):
    client = TestClient(_app(tmp_path))
    r = client.post("/sessions", json={"autonomy": "guarded", "message": "train"})
    tid = r.headers["x-thread-id"]
    events = list(_sse_events(r))
    confirms = [e for e in events if e["event"] == "confirm"]
    assert confirms and "interrupt" in confirms[-1]["data"]
    iid = confirms[-1]["data"]["interrupt"]
    r2 = client.post(f"/sessions/{tid}/resume", json={"answers": {iid: "yes"}})
    assert any(e["event"] == "message" and "Trained" in e["data"].get("text", "") for e in _sse_events(r2))


def test_unknown_thread_404(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/sessions/nope").status_code == 404
    assert client.post("/sessions/nope/resume", json={"answer": "y"}).status_code == 404
    assert client.post("/sessions/nope/message", json={"message": "x"}).status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL (`create_app` is the Task-3 stub raising `NotImplementedError`).

- [ ] **Step 3: Implement the API**

```python
"""FastAPI/SSE surface over the V2 agent. SSE out, POST in.

Two mechanisms, mirrored as two endpoints (never overloaded): a new conversation
turn is a fresh invoke with a HumanMessage (/message), and a confirmation reply is
a Command(resume=...) against pending interrupt ids (/resume). A session's
autonomy is set once at start and persists in checkpointed state.
"""
from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ..agents.agent import build_agent
from ..agents.training import run_retraining, run_training
from ..config import get_settings
from ..llm.provider import get_chat_model


def _default_factory(checkpointer):
    return build_agent(
        chat_model=get_chat_model("smart"),
        train_fn=run_training, retrain_fn=run_retraining,
        tools_chat_model=get_chat_model("cheap"),
        ticket_dir="artifacts/tickets", models_dir="artifacts/models",
        checkpointer=checkpointer,
    )


def _sse(event: str, data) -> str:
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


def create_app(agent_factory=None, checkpointer=None) -> FastAPI:
    app = FastAPI(title="Sentinel V2")
    factory = agent_factory or _default_factory
    agent = factory(checkpointer)

    def _thread(tid: str) -> dict:
        return {"configurable": {"thread_id": tid}}

    def _require(thread: dict) -> None:
        if agent.get_state(thread).created_at is None:
            raise HTTPException(status_code=404, detail="unknown session")

    def _stream(inp, thread):
        try:
            for mode, chunk in agent.stream(inp, thread, stream_mode=["custom", "updates"]):
                if mode == "custom":
                    yield _sse(chunk.get("type", "notify"), chunk)
                elif mode == "updates":
                    for _node, upd in (chunk or {}).items():
                        for m in (upd.get("messages") if isinstance(upd, dict) else None) or []:
                            if isinstance(m, AIMessage) and getattr(m, "content", ""):
                                yield _sse("message", {"text": m.content})
                            for tc in getattr(m, "tool_calls", None) or []:
                                yield _sse("tool_call", {"name": tc["name"], "args": tc["args"]})
        except Exception as exc:  # noqa: BLE001 - surface as event, not truncation
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return
        st = agent.get_state(thread)
        if st.interrupts:
            for it in st.interrupts:
                yield _sse("confirm", {**it.value, "interrupt": it.id})
        else:
            yield _sse("done", {})

    @app.post("/sessions")
    def start(body: dict | None = None):
        body = body or {}
        tid = uuid.uuid4().hex
        thread = _thread(tid)
        autonomy = body.get("autonomy") or get_settings().sentinel_autonomy
        inp = {"messages": [HumanMessage(body.get("message", "Hello"))], "autonomy": autonomy}
        return StreamingResponse(_stream(inp, thread), media_type="text/event-stream",
                                 headers={"x-thread-id": tid})

    @app.post("/sessions/{tid}/message")
    def message(tid: str, body: dict):
        thread = _thread(tid)
        _require(thread)
        inp = {"messages": [HumanMessage(body["message"])]}
        return StreamingResponse(_stream(inp, thread), media_type="text/event-stream",
                                 headers={"x-thread-id": tid})

    @app.post("/sessions/{tid}/resume")
    def resume(tid: str, body: dict):
        thread = _thread(tid)
        _require(thread)
        answers = body.get("answers")
        if answers is None:
            st = agent.get_state(thread)
            if len(st.interrupts) != 1:
                raise HTTPException(status_code=400, detail="multiple confirmations pending; use 'answers' map")
            answers = {st.interrupts[0].id: body.get("answer", "")}
        return StreamingResponse(_stream(Command(resume=answers), thread),
                                 media_type="text/event-stream", headers={"x-thread-id": tid})

    @app.get("/sessions/{tid}")
    def snapshot(tid: str):
        thread = _thread(tid)
        _require(thread)
        st = agent.get_state(thread)
        msgs = st.values.get("messages", [])
        return {
            "autonomy": st.values.get("autonomy"),
            "pending_confirmations": [{"interrupt": it.id, **it.value} for it in st.interrupts],
            "last_message": msgs[-1].content if msgs else None,
        }

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest
uv run ruff check sentinel/api/app.py tests/test_api.py
git add -A
git commit -m "feat(api): HTTP/SSE surface over the V2 agent (message/resume/confirm, autonomy)"
```

---

## Task 9: README + learning note

**Files:**
- Modify: `README.md`
- Create: `docs/learning/04-agentic-ds.md`

**Interfaces:** none (documentation).

Written LAST, per the project's learning-docs practice, so it reflects the real implementation. Not before the code is green.

- [ ] **Step 1: Update `README.md`** - replace the "Agent layer (M2)" run instructions with the V2 agent: the `create_agent` hub, the tool list, the registry, guarded vs `--autonomous`, and the new API endpoints (`/sessions`, `/sessions/{id}/message`, `/sessions/{id}/resume`). Keep the `curl -N` streaming note. State the reference training result is unchanged.

- [ ] **Step 2: Write `docs/learning/04-agentic-ds.md`** - a learning note covering: why the event-router became a reasoning agent; what LangChain's `create_agent` gives you and the custom-state-schema pattern (subclass `langchain.agents.middleware.AgentState`, add `autonomy`); the `InjectedState` + checkpointed-autonomy pattern; why guarded tools return strings instead of raising (the tool node re-raises non-validation exceptions); the registry as the single source of truth; and the id-addressed multi-confirmation resume. Include 3-4 exercises (e.g. "add a `list_tickets` read-only tool", "make `train` guarded only on the first run of a session", "add a budget rail that blocks a 3rd retrain"). Match the style and depth of `docs/learning/03-resumable-and-api.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/learning/04-agentic-ds.md
git commit -m "docs(v2): README + learning note for the agentic data scientist"
```

---

## Self-Review

**Spec coverage:** create_agent hub (T6); chat-model seam + deps (T4); state shrink + DSAgentState over middleware.AgentState (T6); all 10 tools + return-not-raise (T5); train_one (T2); registry incl. readings + active-delete guard (T1); comparable-metrics rule (T5 compare); guarded set incl. run_monitor + autonomous mode + auto_approved (T5); autonomy in checkpointed state via InjectedState (T5/T6); confirm returns-not-raises (T5); multi-confirmation id-addressed resume (T6 test, T7 CLI, T8 API); CLI + HTTP surfaces with /message vs /resume (T7, T8); teardown of V1 graph/interviewer (T3); offline tests incl. FakeChatModel, per-tool, registry round-trip, comparable-metrics, rail, batched-confirmation, autonomy-persistence, e2e trajectory (T1/T5/T6); learning note (T9). All spec sections map to a task.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step has real code; every test step has real assertions. Two explicit "adjust the tool-invocation path if the installed LangChain's InjectedState test-calling differs" notes (T5 Step 4, T6 Step 4) are deliberate resilience instructions, not placeholders - the assertions are fixed.

**Type consistency:** `Registry` method names match across T1 and their uses in T5/T7/T8 (`register`, `get`, `provenance`, `readings`, `list`, `active`, `set_active`, `remove`, `load_predict`). `make_tools(*, train_fn, retrain_fn, chat_model, ticket_dir, registry)` matches its call in `build_agent` (T6) where `chat_model=tools_chat_model`. `build_agent(*, chat_model, train_fn, retrain_fn, tools_chat_model, ticket_dir, models_dir, checkpointer)` matches its calls in T6/T7/T8. `run_training(config)` and `run_retraining(model_id, hyperparameters, rul_cap, window)` match the `train_fn`/`retrain_fn` shapes tools expect. `write_report(result, chat_model, config, best_model_name)` matches T4 and the T5 write_report tool. SSE event names match the spec vocabulary.
