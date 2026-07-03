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

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..core import automl, data, features
from ..pipeline import set_seeds
from .state import InterviewConfig


@dataclass
class TrainingRun:
    """What one training run hands to the rest of the graph.

    result: the M1 `TrainResult` (leaderboard + best model + metrics).
    test_eval: one row per FD001 test unit at its last cycle, with true RUL -
        the "incoming readings" the monitor steps through.
    predict: maps a feature frame to predicted RUL (built from the saved model).
    """

    result: automl.TrainResult
    test_eval: pd.DataFrame
    predict: Callable[[pd.DataFrame], "pd.Series | list[float]"]


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
    )

    # Load the persisted preprocessing+model pipeline so the monitor can predict
    # standalone, without re-entering the PyCaret experiment context.
    from pycaret.regression import load_model, predict_model

    model = load_model(str(result.model_path.with_suffix("")))

    def predict(frame: pd.DataFrame):
        preds = predict_model(model, data=frame)
        return preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]

    return TrainingRun(result=result, test_eval=test_eval, predict=predict)
