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
from typing import Callable

import pandas as pd
from pycaret.regression import (
    create_model,
    finalize_model,
    models,
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


def _rank_models(results: list[tuple[str, dict, object, str]]) -> tuple[pd.DataFrame, object, str]:
    """Rank per-model CV results by RMSE (lower is better) and pick the winner.

    `results` is one `(friendly_name, cv_metrics, fitted_model, model_id)` per
    trained model. Returns the leaderboard frame (best first - each row carries the
    retrainable PyCaret `id` and the friendly `Model` name, so runner-up models can
    be retrained by id, not just admired by name), the winning model, and its name.
    Pure - no PyCaret - so the ranking that decides the winner is unit-testable.
    """
    ordered = sorted(results, key=lambda r: r[1]["RMSE"])
    leaderboard = pd.DataFrame(
        [{"id": model_id, "Model": name, **metrics} for name, metrics, _, model_id in ordered]
    )
    best_name, _, best_model, _ = ordered[0]
    return leaderboard, best_model, best_name


def train_and_evaluate(
    train_df: pd.DataFrame,
    target: str,
    test_df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    ignore_features: list[str] | None = None,
    include: list[str] | None = DEFAULT_MODELS,
    session_id: int = 42,
    fold: int = 3,
    on_model_start: Callable[[str, int, int], None] | None = None,
    on_model_end: Callable[[str, int, int, dict], None] | None = None,
    on_stage: Callable[..., None] | None = None,
) -> TrainResult:
    """Set up PyCaret, train each candidate model, finalize the best, evaluate + persist.

    - `ignore_features`: kept in the frame but excluded from the model (e.g.
      the `unit` identifier and raw `cycle`).
    - Each candidate is trained one at a time (`create_model`) rather than in a
      single opaque `compare_models` call, so two PyTorch-style lifecycle hooks can
      report progress: `on_model_start(name, index, total)` fires just before a
      model is trained, `on_model_end(name, index, total, cv_metrics)` right after.
      `index` is 1-based, `total` is the candidate count. Selection is still by
      cross-validated RMSE - identical winner to `compare_models(sort="RMSE")`.
    - `on_stage(stage, detail="")` marks the coarse phases around the model loop
      (`winner_selected` with the winner's name, `evaluating`, `saving`) so the
      seconds between the last model and the finished run are not silent.
    - Evaluation is on `test_df` (the real FD001 test set), not just CV folds.
    - Saves the finalized model and a `metrics.json` under `artifacts_dir`.
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
        n_jobs=1,          # single-threaded -> deterministic ordering
        verbose=False,
    )

    name_of = models()["Name"].to_dict()  # model id -> friendly name ("et" -> "Extra Trees Regressor")
    candidate_ids = list(include) if include is not None else list(name_of)

    results: list[tuple[str, dict, object, str]] = []
    total = len(candidate_ids)
    for index, model_id in enumerate(candidate_ids, start=1):
        name = name_of.get(model_id, str(model_id))
        if on_model_start is not None:
            on_model_start(name, index, total)  # announce before the slow CV, so the UI isn't blank
        try:
            model = create_model(model_id, verbose=False)  # trains + cross-validates
        except Exception:  # noqa: BLE001 - a single unavailable model must not sink the run
            continue
        mean = pull().loc["Mean"]  # fold-averaged CV metrics for this model
        cv_metrics = {col: float(mean[col]) for col in mean.index if pd.notna(mean[col])}
        results.append((name, cv_metrics, model, model_id))
        if on_model_end is not None:
            on_model_end(
                name, index, total,
                {"rmse": cv_metrics.get("RMSE"), "mae": cv_metrics.get("MAE"), "r2": cv_metrics.get("R2")},
            )

    leaderboard, best_model, best_name = _rank_models(results)

    _stage("winner_selected", best_name)
    final_model = finalize_model(best_model)  # refit on the full training data

    # Held-out evaluation on the real FD001 test set.
    _stage("evaluating")
    preds = predict_model(final_model, data=test_df)
    y_pred = preds["prediction_label"] if "prediction_label" in preds else preds.iloc[:, -1]
    metrics = _regression_metrics(test_df[target], y_pred)

    _stage("saving")
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
    """Train one named model with explicit hyperparameters and persist it."""

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
    model = create_model(model_id, **hyperparameters)
    final_model = finalize_model(model)

    _stage("evaluating")
    preds = predict_model(final_model, data=test_df)
    y_pred = (
        preds["prediction_label"]
        if "prediction_label" in preds
        else preds.iloc[:, -1]
    )
    metrics = _regression_metrics(test_df[target], y_pred)

    _stage("saving")
    model_path = artifacts_dir / f"retrain_{model_id}"
    save_model(final_model, str(model_path))

    leaderboard = pd.DataFrame([{"Model": model_id, **metrics}])
    return TrainResult(
        leaderboard=leaderboard,
        best_model=final_model,
        metrics=metrics,
        model_path=model_path.with_suffix(".pkl"),
        metrics_path=artifacts_dir / f"retrain_{model_id}_metrics.json",
    )
