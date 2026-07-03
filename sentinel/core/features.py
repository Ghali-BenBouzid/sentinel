"""Rolling-window feature engineering.

A raw C-MAPSS row is a single instant. What actually predicts failure is the
*trend* of each sensor over recent cycles. So per engine unit, per sensor, we
compute rolling statistics over the last N cycles (mean, std, slope) and turn
the time series into a flat feature table keyed to the RUL target.

FD001 also ships several sensors that never move (constant across the whole
fleet). They carry no signal, so we drop them before featurizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSOR_COLUMNS

DEFAULT_WINDOW = 5


def informative_sensors(
    df: pd.DataFrame,
    sensor_cols: list[str] = SENSOR_COLUMNS,
    std_threshold: float = 1e-6,
) -> list[str]:
    """Return the sensors that actually vary (drop near-zero-variance ones).

    Decided on the *training* frame, then applied to both train and test so the
    two feature tables have identical columns.
    """
    stds = df[sensor_cols].std()
    return [c for c in sensor_cols if stds[c] > std_threshold]


def _window_slope(values: np.ndarray) -> float:
    """Least-squares slope of a value window against cycle position (0..k-1).

    Slope answers "is this sensor rising or falling, and how fast?" - the part
    of a trend a single mean can't capture. Short windows (start of a series)
    just fit fewer points; a lone point has no slope, so 0.
    """
    n = len(values)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    t_dev = t - t.mean()
    denom = (t_dev**2).sum()
    if denom == 0:
        return 0.0
    return float((t_dev * (values - values.mean())).sum() / denom)


def build_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Turn a per-(unit, cycle) sensor series into a rolling-window feature table.

    For each kept sensor we emit `<sensor>_mean`, `<sensor>_std`, `<sensor>_slope`
    over the trailing ``window`` cycles, computed independently per unit (windows
    never bleed across engines). `unit` and `cycle` are carried through for
    joining/eval; a `RUL` column, if present on the input, is carried through as
    the target.
    """
    out = df[["unit", "cycle"]].copy()
    grouped = df.groupby("unit", sort=False)

    for col in sensor_cols:
        roll = grouped[col].rolling(window, min_periods=1)
        # groupby().rolling() indexes by (unit, original_index); drop the unit
        # level so values realign with `out`'s original index.
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True).fillna(0.0)  # std of 1 point -> 0
        slope = roll.apply(_window_slope, raw=True).reset_index(level=0, drop=True)
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = std
        out[f"{col}_slope"] = slope

    if "RUL" in df.columns:
        out["RUL"] = df["RUL"].to_numpy()

    return out


if __name__ == "__main__":
    # Self-check: slope sign/magnitude and per-unit isolation.
    assert _window_slope(np.array([1.0])) == 0.0
    assert _window_slope(np.array([1.0, 2.0, 3.0])) == 1.0  # rises by 1/cycle
    assert _window_slope(np.array([3.0, 1.0])) == -2.0      # falls by 2 over 1 cycle
    demo = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s1": [10.0, 12.0, 14.0, 100.0, 100.0],
            "s2": [5.0, 5.0, 5.0, 5.0, 5.0],  # constant -> dropped
        }
    )
    keep = informative_sensors(demo, sensor_cols=["s1", "s2"])
    assert keep == ["s1"], keep
    feats = build_features(demo, keep, window=2)
    # unit 2 must not inherit unit 1's rising trend.
    u2 = feats[feats["unit"] == 2]
    assert (u2["s1_slope"] == 0.0).all(), u2
    print("features self-check OK; columns:", list(feats.columns))
