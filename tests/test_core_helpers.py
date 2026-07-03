"""Fast, offline unit tests for the pure-Python DS-core helpers.

These exercise the RUL target derivation and rolling-window featurization on
tiny synthetic frames - no network, no PyCaret, no model training.
"""

from __future__ import annotations

import pandas as pd

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
    u1_last_slope = feats[(feats["unit"] == 1) & (feats["cycle"] == 3)]["s1_slope"].iloc[0]
    assert u1_last_slope == 2.0
