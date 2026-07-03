"""PyCaret AutoML wrapper: train, compare, evaluate, persist.

PyCaret does the boilerplate a real model comparison needs - preprocessing,
cross-validated training of a whole shelf of regressors, and a ranked
leaderboard - in a few calls. We drive it deterministically and evaluate the
finalized best model on the held-out FD001 test set (real generalization, not
just CV on train).

This is the seam the agent layer plugs into later: `train_and_evaluate` returns
the leaderboard + fitted model + metrics that the orchestrator/report-writer
will summarize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pycaret.regression import (
    compare_models,
    finalize_model,
    predict_model,
    pull,
    save_model,
    setup,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# A solid spread of model families (linear / regularized / tree / boosting /
# neighbours) - a genuine leaderboard while keeping the run fast and
# reproducible. Pass include=None to compare PyCaret's full shelf instead.
DEFAULT_MODELS = ["lr", "ridge", "lasso", "en", "huber", "knn", "dt", "rf", "et", "gbr", "lightgbm"]


@dataclass
class TrainResult:
    leaderboard: pd.DataFrame
    best_model: object
    metrics: dict[str, float]
    model_path: Path
    metrics_path: Path


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    target: str,
    test_df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    ignore_features: list[str] | None = None,
    include: list[str] | None = DEFAULT_MODELS,
    session_id: int = 42,
    fold: int = 3,
) -> TrainResult:
    """Set up PyCaret, compare models, finalize the best, evaluate + persist.

    - `ignore_features`: kept in the frame but excluded from the model (e.g.
      the `unit` identifier and raw `cycle`).
    - Evaluation is on `test_df` (the real FD001 test set), not just CV folds.
    - Saves the finalized model and a `metrics.json` under `artifacts_dir`.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ignore_features = ignore_features or []

    setup(
        data=train_df,
        target=target,
        session_id=session_id,
        ignore_features=ignore_features,
        fold=fold,
        n_jobs=1,          # single-threaded -> deterministic ordering
        verbose=False,
    )

    best_model = compare_models(include=include, sort="RMSE", verbose=False)
    leaderboard = pull()

    final_model = finalize_model(best_model)  # refit on the full training data

    # Held-out evaluation on the real FD001 test set.
    preds = predict_model(final_model, data=test_df)
    y_pred = preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]
    metrics = _regression_metrics(test_df[target], y_pred)

    model_path = artifacts_dir / "rul_model"
    save_model(final_model, str(model_path))  # writes rul_model.pkl

    summary = {
        "dataset": "C-MAPSS FD001",
        "target": target,
        "best_model": type(best_model).__name__,
        "test_metrics": metrics,
        "leaderboard": leaderboard.reset_index(drop=True).to_dict(orient="records"),
    }
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    leaderboard.to_csv(artifacts_dir / "leaderboard.csv", index=False)

    return TrainResult(
        leaderboard=leaderboard,
        best_model=best_model,
        metrics=metrics,
        model_path=model_path.with_suffix(".pkl"),
        metrics_path=metrics_path,
    )
