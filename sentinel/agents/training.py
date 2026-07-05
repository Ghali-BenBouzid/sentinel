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


# Human-facing text for each coarse training stage. `{detail}` (if present) is
# filled from the stage's detail arg (e.g. the winning model's name).
_STAGE_TEXT = {
    "loading_data": "Loading and preparing the FD001 dataset ...",
    "winner_selected": "Winner selected: {detail} - refitting it on all the training data ...",
    "evaluating": "Evaluating the winning model on the held-out test set ...",
    "saving": "Saving the trained model ...",
    "loading_model": "Loading the saved model to hand to the monitor ...",
}


def _stage_event(stage: str, detail: str = "") -> dict:
    """Build the `stage` custom event for one coarse training phase (pure).

    Carries a machine-readable `stage` id and a human `text`, so a client can drive
    a stepper or just show the line. `detail` (e.g. the winner name) is interpolated
    into the text and echoed as a field when present.
    """
    text = _STAGE_TEXT[stage].format(detail=detail)
    event = {"type": "stage", "stage": stage, "text": text}
    if detail:
        event["detail"] = detail
    return event


def _training_stream():
    """Build `(on_model_start, on_model_end, on_stage)` bound to the active stream writer.

    Each candidate model gets a `model_training` event when PyCaret starts it and a
    `model_trained` event (plus its CV metrics) when it finishes; both carry `index`
    and `total` so a client can render "3 of 11" and a progress bar. `on_stage` emits
    the coarse `stage` events around the loop (data load, winner, evaluation, save,
    model load) so the seconds outside the loop are not silent either.

    Uses the graph's active stream writer; degrades to no-op hooks when there is no
    active stream (direct/CLI-less use), the same seam the interviewer/trainer use.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - no active stream context; run silently
        return (lambda *a: None), (lambda *a: None), (lambda *a: None)

    def on_model_start(name: str, index: int, total: int) -> None:
        writer({"type": "model_training", "name": name, "index": index, "total": total})

    def on_model_end(name: str, index: int, total: int, cv_metrics: dict) -> None:
        writer({"type": "model_trained", "name": name, "index": index, "total": total, "cv_metrics": cv_metrics})

    def on_stage(stage: str, detail: str = "") -> None:
        writer(_stage_event(stage, detail))

    return on_model_start, on_model_end, on_stage


def run_training(config: InterviewConfig, data_dir: str = "data", artifacts_dir: str = "artifacts") -> TrainingRun:
    """Load FD001, featurize, train/evaluate, and build a prediction function."""
    on_model_start, on_model_end, on_stage = _training_stream()

    set_seeds()
    on_stage("loading_data")
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
        on_model_start=on_model_start,
        on_model_end=on_model_end,
        on_stage=on_stage,
    )

    # Load the persisted preprocessing+model pipeline so the monitor can predict
    # standalone, without re-entering the PyCaret experiment context.
    on_stage("loading_model")
    predict = load_predict(str(result.model_path))

    return TrainingRun(result=result, test_eval=test_eval, predict=predict)
