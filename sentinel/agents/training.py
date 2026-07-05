"""Thin wrapper that runs the M1 DS core from an `InterviewConfig`.

This is the seam between the agent layer and Milestone 1: it calls the same
`data` / `features` / `automl` functions `sentinel.pipeline` does, driven by the
knobs the interviewer collected (`window`, `rul_cap`). It returns everything the
downstream nodes need - the `TrainResult` for the report writer, plus the
held-out test rows and a prediction function for the monitor to step through.

The trainer node treats this as a black box, so tests can inject a stub
`train_fn` and never touch PyCaret.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core import automl, data, features
from ..pipeline import set_seeds
from .state import InterviewConfig


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NATIVE Python types.

    `df.to_dict("records")` leaks numpy scalars (np.int64/np.float64) that the
    checkpointer's msgpack serializer rejects; the JSON round-trip forces builtins.
    """
    return json.loads(df.to_json(orient="records"))


def load_predict(model_path: str) -> Callable[[pd.DataFrame], "pd.Series | list[float]"]:
    """Load a persisted PyCaret pipeline and return a frame -> predicted-RUL fn.

    Factored out of `run_training` so the monitor can rehydrate the same
    prediction function from `model_path` after the live closure was dropped at
    the checkpoint boundary. Imports PyCaret lazily so importing this module
    (e.g. from the monitor) stays cheap.
    """
    from pycaret.regression import load_model, predict_model

    model = load_model(str(Path(model_path).with_suffix("")))  # load_model wants no .pkl suffix

    def predict(frame: pd.DataFrame):
        preds = predict_model(model, data=frame)
        return preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]

    return predict


@dataclass
class TrainingRun:
    """What one training run hands to the rest of the graph.

    result: the M1 `TrainResult` (leaderboard + best model + metrics).
    test_eval: one row per FD001 test unit at its last cycle, with true RUL -
        the "incoming readings" the monitor steps through.
    predict: maps a feature frame to predicted RUL (built from the saved model).

    Only `to_state()` crosses the graph-state boundary - the live `predict`
    closure, the DataFrames, and the estimator are not checkpoint-serializable.
    """

    result: automl.TrainResult
    test_eval: pd.DataFrame
    predict: Callable[[pd.DataFrame], "pd.Series | list[float]"]

    def to_state(self) -> dict:
        """Reduce to msgpack-safe native-Python data for the checkpointed state.

        The heavy artifacts are rehydrated downstream: the model from
        `model_path` (via `load_predict`), `test_eval` from its records.
        """
        r = self.result
        lb = r.leaderboard
        best = lb.iloc[0]["Model"] if "Model" in lb.columns else type(r.best_model).__name__
        return {
            "metrics": {k: float(v) for k, v in r.metrics.items()},
            "leaderboard": _records(lb),
            "best_model_name": str(best),
            "model_path": str(r.model_path),
            "test_eval": _records(self.test_eval),
        }


def _model_progress() -> "Callable[[str, dict], None]":
    """Build an `on_model(name, cv_metrics)` that streams one event per trained model.

    Uses the graph's active stream writer so each model PyCaret finishes surfaces
    as a `model_trained` custom event (the long comparison is otherwise silent).
    Degrades to a no-op when there is no active stream (direct/CLI-less use), the
    same seam the interviewer/trainer use for their custom events.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - no active stream context; run silently
        return lambda name, metrics: None

    def on_model(name: str, cv_metrics: dict) -> None:
        writer({"type": "model_trained", "name": name, "cv_metrics": cv_metrics})

    return on_model


def run_training(config: InterviewConfig, data_dir: str = "data", artifacts_dir: str = "artifacts") -> TrainingRun:
    """Load FD001, featurize, train/evaluate, and build a prediction function."""
    set_seeds()
    ds = data.load_fd001(data_dir=data_dir, rul_cap=config.rul_cap)

    keep = features.informative_sensors(ds.train)
    train_feat = features.build_features(ds.train, keep, window=config.window)
    test_feat = features.build_features(ds.test, keep, window=config.window)
    test_eval = data.build_test_eval(test_feat, ds.rul_truth, rul_cap=config.rul_cap)

    result = automl.train_and_evaluate(
        train_feat,
        target="RUL",
        test_df=test_eval,
        artifacts_dir=artifacts_dir,
        ignore_features=["unit", "cycle"],
        on_model=_model_progress(),
    )

    # Load the persisted preprocessing+model pipeline so the monitor can predict
    # standalone, without re-entering the PyCaret experiment context.
    predict = load_predict(str(result.model_path))

    return TrainingRun(result=result, test_eval=test_eval, predict=predict)
