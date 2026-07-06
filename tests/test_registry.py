"""The disk-backed model registry: the single source of truth for models."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _fake_saved_model(tmp_path: Path) -> Path:
    """Stand in for a PyCaret .pkl on disk (register only copies bytes)."""
    p = tmp_path / "src_model.pkl"
    p.write_bytes(b"fake-model-bytes")
    return p


def _reg(tmp_path):
    from sentinel.agents.registry import Registry

    return Registry(tmp_path / "models")


def _register(reg, tmp_path, family="et", rul_cap=125, window=5, rmse=17.1):
    return reg.register(
        family=family,
        model_path=_fake_saved_model(tmp_path),
        metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        leaderboard=[{"Model": "Extra Trees", "RMSE": rmse}],
        provenance={
            "source": "train",
            "model_id": family,
            "hyperparameters": {},
            "config": {"rul_cap": rul_cap, "window": window},
            "parent": None,
        },
        test_eval=[{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}],
    )


def test_register_creates_ids_and_layout_and_first_is_active(tmp_path):
    reg = _reg(tmp_path)
    id1 = _register(reg, tmp_path)
    id2 = _register(reg, tmp_path)
    assert id1 == "et-v1" and id2 == "et-v2"
    assert reg.active() == "et-v1"
    root = tmp_path / "models" / "et-v1"
    assert (root / "model.pkl").read_bytes() == b"fake-model-bytes"
    assert json.loads((root / "metrics.json").read_text())["rmse"] == 17.1
    assert (
        json.loads((root / "provenance.json").read_text())["config"]["rul_cap"]
        == 125
    )
    assert json.loads((root / "readings.json").read_text())[0]["unit"] == 1
    assert set(reg.list()) == {"et-v1", "et-v2"}


def test_set_active_and_get_and_readings(tmp_path):
    reg = _reg(tmp_path)
    _register(reg, tmp_path)
    id2 = _register(reg, tmp_path)
    reg.set_active(id2)
    assert reg.active() == id2
    assert reg.get(id2)["metrics"]["rmse"] == 17.1
    assert reg.readings(id2)[0]["RUL"] == 40.0
    with pytest.raises(KeyError):
        reg.get("nope")
    with pytest.raises(KeyError):
        reg.set_active("nope")


def test_remove_refuses_active_and_deletes_inactive(tmp_path):
    reg = _reg(tmp_path)
    a = _register(reg, tmp_path)
    b = _register(reg, tmp_path)
    with pytest.raises(ValueError):
        reg.remove(a)
    reg.remove(b)
    assert reg.list() == [a]
    assert not (tmp_path / "models" / b).exists()


def test_load_predict_rehydrates_via_pycaret(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    mid = _register(reg, tmp_path)
    import sentinel.agents.registry as R

    monkeypatch.setattr(
        R, "_pycaret_load_predict", lambda pkl_path: (lambda frame: [50.0] * len(frame))
    )
    predict = reg.load_predict(mid)
    assert predict(pd.DataFrame([{"s2": 1.0}, {"s2": 2.0}])) == [50.0, 50.0]
