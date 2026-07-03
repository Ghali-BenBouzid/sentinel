"""End-to-end M1 pipeline: FD001 -> features -> AutoML -> saved model + metrics.

Run it:  python -m sentinel.pipeline

Steps:
  1. Download/load the real NASA C-MAPSS FD001 data (cached under ./data).
  2. Drop dead sensors, build rolling-window features for train and test.
  3. Assemble the eval frame (last cycle per test unit + true RUL).
  4. PyCaret: compare models, finalize the best, evaluate on the test set.
  5. Save the model + metrics under ./artifacts and print a summary.
"""

from __future__ import annotations

import random

import numpy as np

from .core import automl, data, features

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Pin RNGs so a run is reproducible (PyCaret gets its own session_id)."""
    random.seed(seed)
    np.random.seed(seed)


def run(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts",
    window: int = features.DEFAULT_WINDOW,
    rul_cap: int = data.DEFAULT_RUL_CAP,
) -> automl.TrainResult:
    set_seeds()

    print("[1/4] loading FD001 ...")
    ds = data.load_fd001(data_dir=data_dir, rul_cap=rul_cap)
    print(f"      train {ds.train.shape}, test {ds.test.shape}, rul_cap={rul_cap}")

    print("[2/4] building rolling-window features ...")
    keep = features.informative_sensors(ds.train)
    print(f"      kept {len(keep)}/{len(data.SENSOR_COLUMNS)} sensors: {keep}")
    train_feat = features.build_features(ds.train, keep, window=window)
    test_feat = features.build_features(ds.test, keep, window=window)
    test_eval = data.build_test_eval(test_feat, ds.rul_truth, rul_cap=rul_cap)
    print(f"      train_feat {train_feat.shape}, test_eval {test_eval.shape}")

    print("[3/4] AutoML: comparing models (this is the slow step) ...")
    result = automl.train_and_evaluate(
        train_feat,
        target="RUL",
        test_df=test_eval,
        artifacts_dir=artifacts_dir,
        ignore_features=["unit", "cycle"],  # identifiers, not signal
        session_id=SEED,
    )

    print("[4/4] done.\n")
    print("Leaderboard (top of comparison):")
    cols = [c for c in ["Model", "MAE", "RMSE", "R2"] if c in result.leaderboard.columns]
    print(result.leaderboard[cols].head(len(result.leaderboard)).to_string())
    m = result.metrics
    print(f"\nBest model: {type(result.best_model).__name__}")
    print(f"Held-out FD001 test:  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.3f}")
    print(f"Saved model:   {result.model_path}")
    print(f"Saved metrics: {result.metrics_path}")
    return result


if __name__ == "__main__":
    run()
