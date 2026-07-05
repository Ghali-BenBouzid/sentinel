"""Fast, offline unit tests for the pure-Python DS-core helpers.

These exercise the RUL target derivation and rolling-window featurization on
tiny synthetic frames - no network, no PyCaret, no model training.
"""

from __future__ import annotations

import pandas as pd

from sentinel.core.automl import _rank_models
from sentinel.core.data import add_rul
from sentinel.core.features import build_features, informative_sensors


def test_add_rul_counts_down_to_failure_and_caps():
    # Unit 1 runs 4 cycles (last cycle = failure), unit 2 runs 2 cycles.
    df = pd.DataFrame({"unit": [1, 1, 1, 1, 2, 2], "cycle": [1, 2, 3, 4, 1, 2]})

    ruls = add_rul(df, rul_cap=None)["RUL"].tolist()
    # RUL = last cycle of the unit - this cycle.
    assert ruls == [3, 2, 1, 0, 1, 0]

    # Cap clips large early-life RUL, leaves small values untouched.
    capped = add_rul(df, rul_cap=2)["RUL"].tolist()
    assert capped == [2, 2, 1, 0, 1, 0]


def test_build_features_shape_and_per_unit_isolation():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s1": [10.0, 12.0, 14.0, 100.0, 100.0],  # unit 1 rising, unit 2 flat
            "s2": [5.0, 5.0, 5.0, 5.0, 5.0],  # constant -> not informative
        }
    )

    keep = informative_sensors(df, sensor_cols=["s1", "s2"])
    assert keep == ["s1"]

    feats = build_features(df, keep, window=2)
    # unit + cycle carried through, plus mean/std/slope for the one kept sensor.
    assert list(feats.columns) == ["unit", "cycle", "s1_mean", "s1_std", "s1_slope"]
    assert len(feats) == len(df)

    # Unit 2 is flat, so its window slope must be 0 (no bleed from unit 1's trend).
    u2 = feats[feats["unit"] == 2]
    assert (u2["s1_slope"] == 0.0).all()
    # Unit 1 rises 2 units/cycle -> trailing 2-cycle slope is 2.0.
    u1_last_slope = feats[(feats["unit"] == 1) & (feats["cycle"] == 3)][
        "s1_slope"
    ].iloc[0]
    assert u1_last_slope == 2.0


def test_rank_models_orders_by_rmse_and_picks_lowest():
    # Winner is decided by cross-validated RMSE (lower is better), regardless of
    # the order models were trained in - this is the selection contract that
    # replaced compare_models(sort="RMSE").
    et = object()
    results = [
        ("Linear Regression", {"RMSE": 25.0, "MAE": 20.0, "R2": 0.4}, object()),
        ("Extra Trees Regressor", {"RMSE": 17.1, "MAE": 12.0, "R2": 0.82}, et),
        ("LightGBM", {"RMSE": 18.4, "MAE": 13.0, "R2": 0.79}, object()),
    ]

    leaderboard, best_model, best_name = _rank_models(results)

    assert best_name == "Extra Trees Regressor"
    assert best_model is et  # the actual fitted object, not a copy
    # Leaderboard is best-first and carries the friendly name + its metrics.
    assert leaderboard["Model"].tolist() == ["Extra Trees Regressor", "LightGBM", "Linear Regression"]
    assert leaderboard.iloc[0]["RMSE"] == 17.1
