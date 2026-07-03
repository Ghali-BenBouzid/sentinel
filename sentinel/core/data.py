"""C-MAPSS FD001 loader.

Downloads (and caches) the NASA C-MAPSS FD001 turbofan run-to-failure data,
reads it into tidy DataFrames, and derives the Remaining Useful Life (RUL)
target used for regression.

Only the FD001 loader is implemented for M1. The V1 goal is to be
dataset-agnostic; the seam for that is the shape of `load_fd001` (returns a
small `FD001` bundle) and the module-level column constants - a future
`load_ai4i(...)` / registry can follow the same pattern. We deliberately do NOT
build the registry machinery yet (YAGNI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# --- Dataset schema -------------------------------------------------------
# Raw C-MAPSS rows are whitespace-delimited: engine unit id, operating cycle,
# 3 operational settings, then 21 sensor channels.
INDEX_COLUMNS = ["unit", "cycle"]
OP_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLUMNS + OP_COLUMNS + SENSOR_COLUMNS

# Standard piecewise-linear RUL cap for C-MAPSS. Early in an engine's life the
# true RUL is large but not meaningfully predictable from sensors (nothing has
# degraded yet), so the community clips the target: below the knee RUL is
# linear, above it we treat the engine as "healthy" at a constant value.
DEFAULT_RUL_CAP = 125

# Public mirror of the original NASA C-MAPSS files (the NASA PCoE portal is not
# a stable direct-download host). Verified to serve the real FD001 files.
_MIRROR_BASE = (
    "https://raw.githubusercontent.com/hankroark/"
    "Turbofan-Engine-Degradation/master/CMAPSSData"
)
_FILES = {
    "train": "train_FD001.txt",
    "test": "test_FD001.txt",
    "rul": "RUL_FD001.txt",
}


@dataclass
class FD001:
    """A loaded FD001 dataset.

    train: full run-to-failure series, one row per (unit, cycle), with the
        derived, capped ``RUL`` target column.
    test: truncated series (engines stopped before failure), no target.
    rul_truth: true RUL at each test unit's *last observed* cycle, indexed by
        unit (uncapped, as shipped by NASA).
    rul_cap: the cap applied to the train target (kept so the same cap can be
        applied consistently when building the test eval labels).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    rul_truth: pd.Series
    rul_cap: int


def download_fd001(data_dir: Path) -> dict[str, Path]:
    """Fetch the three FD001 files into ``data_dir``, caching on disk.

    Returns a mapping name -> local path. Idempotent: already-cached files are
    not re-downloaded.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, filename in _FILES.items():
        dest = data_dir / filename
        if not dest.exists():
            url = f"{_MIRROR_BASE}/{filename}"
            print(f"[data] downloading {filename} ...")
            urlretrieve(url, dest)
        paths[name] = dest
    return paths


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS series file into a tidy frame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some mirror copies carry trailing whitespace -> phantom trailing columns.
    # Keep only the 26 real columns and name them.
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def add_rul(df: pd.DataFrame, rul_cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add the RUL target to a *training* frame.

    RUL for a row = (last cycle that unit reached) - (this row's cycle). Because
    training engines are run to failure, the last cycle is the failure point, so
    this is exactly "cycles remaining until failure". Optionally clipped at
    ``rul_cap`` (see DEFAULT_RUL_CAP).
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - out["cycle"]
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)
    out["RUL"] = rul
    return out


def build_test_eval(
    features: pd.DataFrame,
    rul_truth: pd.Series,
    rul_cap: int | None = DEFAULT_RUL_CAP,
) -> pd.DataFrame:
    """Build an eval-ready frame: one row per test unit at its last cycle.

    ``features`` is any per-(unit, cycle) frame (typically the featurized test
    series). We keep the last observed cycle of each unit and attach the true
    RUL from ``rul_truth``, capping it the same way the training target was so
    metrics compare like with like.
    """
    last = (
        features.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    rul = rul_truth.reindex(last["unit"]).to_numpy()
    if rul_cap is not None:
        rul = rul.clip(max=rul_cap)
    last["RUL"] = rul
    return last


def load_fd001(
    data_dir: str | Path = "data",
    rul_cap: int | None = DEFAULT_RUL_CAP,
) -> FD001:
    """Download (if needed), read, and assemble the FD001 dataset."""
    paths = download_fd001(Path(data_dir))

    train = add_rul(_read_raw(paths["train"]), rul_cap=rul_cap)
    test = _read_raw(paths["test"])

    rul_values = pd.read_csv(paths["rul"], sep=r"\s+", header=None).iloc[:, 0]
    # RUL_FD001 lists the truth for units 1..N in order.
    rul_truth = pd.Series(
        rul_values.to_numpy(),
        index=pd.RangeIndex(1, len(rul_values) + 1, name="unit"),
        name="RUL",
    )

    _verify(train, test, rul_truth)
    return FD001(train=train, test=test, rul_truth=rul_truth, rul_cap=rul_cap or 0)


def _verify(train: pd.DataFrame, test: pd.DataFrame, rul_truth: pd.Series) -> None:
    """Fail loudly if the downloaded data is not shaped like real FD001."""
    expected = set(COLUMNS)
    assert expected.issubset(train.columns), "train missing expected C-MAPSS columns"
    assert expected.issubset(test.columns), "test missing expected C-MAPSS columns"
    # FD001 has 100 train and 100 test engines.
    assert train["unit"].nunique() == 100, f"expected 100 train units, got {train['unit'].nunique()}"
    assert test["unit"].nunique() == 100, f"expected 100 test units, got {test['unit'].nunique()}"
    assert len(rul_truth) == test["unit"].nunique(), "one RUL truth per test unit expected"


if __name__ == "__main__":
    # Smoke check: load real FD001 and print a shape summary.
    ds = load_fd001()
    print("train:", ds.train.shape, "| test:", ds.test.shape)
    print("RUL target range (capped):", ds.train["RUL"].min(), "->", ds.train["RUL"].max())
    print("test units:", ds.test["unit"].nunique(), "| rul_truth:", len(ds.rul_truth))
